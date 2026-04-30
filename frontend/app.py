import sys
import os
import json
import time

import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from backend.database import get_session, engine
from backend.models import Base, User, Appointment
from backend.agent import build_agent_executor
from langchain_classic.memory import ConversationBufferMemory

Base.metadata.create_all(bind=engine)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medica",
    page_icon="🏥",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .app-header { text-align:center; padding:2rem 0 1rem 0; }
    .app-header h1 { font-size:2.2rem; margin-bottom:0.2rem; }
    .app-header p  { color:#94a3b8; font-size:0.95rem; }
    .priority-badge {
        display:inline-block; padding:5px 14px; border-radius:20px;
        font-weight:700; font-size:0.85rem;
    }
    .p0{background:#fee2e2;color:#991b1b;}
    .p1{background:#fef3c7;color:#92400e;}
    .p2{background:#dbeafe;color:#1e40af;}
    .p3{background:#dcfce7;color:#166534;}
    .field-label {
        color:#94a3b8; font-size:0.78rem; font-weight:700;
        text-transform:uppercase; letter-spacing:0.08em;
        margin-bottom:2px;
    }
    .field-value {
        color:#f1f5f9; font-size:1.05rem;
        margin-bottom:1.1rem; font-weight:500;
    }
    .think-step  { border-left:3px solid #334155; padding-left:0.75rem; margin-bottom:0.6rem; }
    .think-thought { border-left-color:#a78bfa; }
    .think-action  { border-left-color:#60a5fa; }
    .think-obs     { border-left-color:#34d399; }
    .think-label   { font-size:0.72rem; font-weight:700; text-transform:uppercase;
                     letter-spacing:0.06em; margin-bottom:2px; }
    .lbl-thought{color:#a78bfa;} .lbl-action{color:#60a5fa;} .lbl-obs{color:#34d399;}
</style>
""", unsafe_allow_html=True)


# ── Thinking callback ─────────────────────────────────────────────────────────
class ThinkingCapture(BaseCallbackHandler):
    """Captures every Thought / Action / Observation from the ReAct loop."""

    def __init__(self):
        self.steps: list[dict] = []

    def on_agent_action(self, action, **_):
        # action.log contains "Thought: ...\nAction: ...\nAction Input: ..."
        thought = ""
        log = action.log or ""
        if "Thought:" in log:
            thought = log.split("Action:")[0].replace("Thought:", "").strip()
        self.steps.append({
            "type": "action",
            "thought": thought,
            "tool": action.tool,
            "tool_input": action.tool_input,
        })

    def on_tool_end(self, output, **_):
        try:
            parsed = json.loads(output) if isinstance(output, str) else output
            display = json.dumps(parsed, indent=2)
        except Exception:
            display = str(output)
        self.steps.append({"type": "observation", "output": display})

    def on_agent_finish(self, finish, **_):
        log = finish.log or ""
        thought = log.replace("Final Answer:", "").strip()
        if thought:
            self.steps.append({"type": "final_thought", "thought": thought})


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "screen": "email",
        "email": "",
        "executor": None,
        "messages": [],       # list of {role, content, thinking?}
        "appointment_id": None,
        "booking_done": False,
        "baseline_appt_id": None,  # highest appt ID that existed before this chat session
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

PRIORITY_LABELS = {0: "Urgent", 1: "High", 2: "Moderate", 3: "Routine"}
PRIORITY_CLASS  = {0: "p0", 1: "p1", 2: "p2", 3: "p3"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _go(screen):
    st.session_state.screen = screen
    st.rerun()


def _get_or_create_executor():
    if st.session_state.executor is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=False, input_key="input",
        )
        st.session_state.executor = build_agent_executor(memory)
    return st.session_state.executor


def _run_agent(message: str) -> tuple[str, list[dict]]:
    """Returns (response_text, thinking_steps)."""
    executor = _get_or_create_executor()
    capture = ThinkingCapture()
    input_text = f"Patient email: {st.session_state.email}\n\nPatient message: {message}"
    try:
        result = executor.invoke(
            {"input": input_text},
            config={"callbacks": [capture]},
        )
        response = result.get("output", "I'm sorry, something went wrong. Please try again.")
    except Exception as e:
        response = f"An error occurred: {str(e)}"
    return response, capture.steps


def _set_baseline():
    """Snapshot the highest existing appointment ID before this chat session starts."""
    if st.session_state.baseline_appt_id is not None:
        return  # already set for this session
    db = get_session()
    try:
        user = db.query(User).filter(User.email == st.session_state.email).first()
        if user and user.appointments:
            st.session_state.baseline_appt_id = max(a.id for a in user.appointments)
        else:
            st.session_state.baseline_appt_id = 0  # no prior appointments
    finally:
        db.close()




def _detect_booking_complete() -> int | None:
    """Return the new appointment ID only if it was created during THIS session."""
    baseline = st.session_state.baseline_appt_id or 0
    db = get_session()
    try:
        user = db.query(User).filter(User.email == st.session_state.email).first()
        if not user or not user.appointments:
            return None
        new_appts = [a for a in user.appointments if a.id > baseline]
        if not new_appts:
            return None
        latest = sorted(new_appts, key=lambda a: a.id)[-1]
        return latest.id if latest.medical_report is not None else None
    finally:
        db.close()


def _fetch_appointment(appt_id: int):
    db = get_session()
    try:
        appt = db.query(Appointment).filter(Appointment.id == appt_id).first()
        if not appt:
            return None
        result = {
            "appointment_id": appt.id,
            "doctor_name": appt.doctor.name,
            "specialty": appt.doctor.specialty,
            "department": appt.doctor.department,
            "slot_datetime": appt.slot.slot_datetime.strftime("%A, %d %B %Y at %H:%M"),
            "priority": appt.priority,
            "patient_email": appt.user_email,
        }
        if appt.medical_report:
            result["summary"] = appt.medical_report.summary
            result["medications"] = appt.medical_report.medication_recommendations
        return result
    finally:
        db.close()


def _render_thinking(steps: list[dict]):
    """Render the ReAct trace inside an expander."""
    if not steps:
        return
    with st.expander("🧠 Agent reasoning trace", expanded=False):
        for step in steps:
            if step["type"] == "action":
                if step["thought"]:
                    st.markdown(
                        f'<div class="think-step think-thought">'
                        f'<div class="think-label lbl-thought">💭 Thought</div>'
                        f'<small>{step["thought"]}</small></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="think-step think-action">'
                    f'<div class="think-label lbl-action">⚡ Action → {step["tool"]}</div>'
                    f'<small><code>{step["tool_input"]}</code></small></div>',
                    unsafe_allow_html=True,
                )
            elif step["type"] == "observation":
                st.markdown(
                    f'<div class="think-step think-obs">'
                    f'<div class="think-label lbl-obs">👁 Observation</div></div>',
                    unsafe_allow_html=True,
                )
                st.code(step["output"], language="json")
            elif step["type"] == "final_thought":
                st.markdown(
                    f'<div class="think-step think-thought">'
                    f'<div class="think-label lbl-thought">✅ Final Thought</div>'
                    f'<small>{step["final_thought"] if "final_thought" in step else step.get("thought","")}</small></div>',
                    unsafe_allow_html=True,
                )


def _render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("thinking"):
                _render_thinking(msg["thinking"])


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🏥 Medica</h1>
    <p>AI-powered patient intake — preliminary recommendations pending physician review</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Email
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "email":
    st.subheader("Welcome — let's get started")
    st.write("Please enter your email address to begin.")

    email = st.text_input("Email address", placeholder="you@example.com", key="email_input")

    if st.button("Continue", type="primary", use_container_width=True):
        if not email or "@" not in email:
            st.error("Please enter a valid email address.")
        else:
            st.session_state.email = email.strip().lower()
            db = get_session()
            user = db.query(User).filter(User.email == st.session_state.email).first()
            db.close()
            _go("chat" if user else "register")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Registration
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "register":
    st.subheader("Create your profile")
    st.write(f"First visit for **{st.session_state.email}**. Please fill in a few details.")

    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    history = st.text_area(
        "Chronic disease history",
        placeholder="e.g. Type 2 diabetes, hypertension — or leave blank if none",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", use_container_width=True):
            _go("email")
    with col2:
        if st.button("Register & Continue", type="primary", use_container_width=True):
            db = get_session()
            db.add(User(
                email=st.session_state.email,
                age=int(age),
                chronic_disease_history=history.strip() or "None",
            ))
            db.commit()
            db.close()
            st.success("Profile created!")
            time.sleep(0.8)
            _go("chat")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Chat
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "chat":
    _set_baseline()  # snapshot pre-existing appointments so we don't false-positive on old ones

    db = get_session()
    user = db.query(User).filter(User.email == st.session_state.email).first()
    db.close()

    with st.sidebar:
        st.markdown("### Patient Info")
        st.write(f"**Email:** {st.session_state.email}")
        if user:
            st.write(f"**Age:** {user.age}")
            st.write(f"**Chronic conditions:** {user.chronic_disease_history}")
        st.divider()
        st.caption("The AI assistant will ask follow-up questions, then book your appointment automatically.")
        if st.button("Start over", use_container_width=True):
            for k in ["screen", "email", "executor", "messages", "appointment_id", "booking_done"]:
                if k == "screen":   st.session_state[k] = "email"
                elif k == "messages": st.session_state[k] = []
                elif k == "booking_done": st.session_state[k] = False
                else: st.session_state[k] = None
            st.rerun()

    st.subheader("Tell us about your symptoms")

    # Welcome message on first open
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("Hello! I'm your hospital intake assistant. Please describe your symptoms and I'll help you get the right appointment. What brings you in today?")

    # Render full chat history (with thinking traces)
    _render_messages()

    # ── "View Appointment" CTA — shown after booking, inside chat ────────────
    if st.session_state.booking_done and st.session_state.appointment_id:
        st.success("✅ Your appointment has been booked and your medical report is ready.")
        if st.button("📋 View Appointment & Medical Report →", type="primary", use_container_width=True):
            _go("confirm")

    # ── Chat input (disabled once booking is done) ────────────────────────────
    elif not st.session_state.booking_done:
        if prompt := st.chat_input("Describe your symptoms..."):
            # Show user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

            # Run agent + capture thinking
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response, thinking = _run_agent(prompt)
                st.markdown(response)
                _render_thinking(thinking)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "thinking": thinking,
            })

            # Check if booking completed — just set flags, don't redirect
            appt_id = _detect_booking_complete()
            if appt_id:
                st.session_state.appointment_id = appt_id
                st.session_state.booking_done = True

            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — Confirmation
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "confirm":
    appt = _fetch_appointment(st.session_state.appointment_id)

    if not appt:
        st.error("Could not load appointment details.")
    else:
        st.success("Your appointment has been booked successfully!")

        priority = appt["priority"]
        badge = f'<span class="priority-badge {PRIORITY_CLASS[priority]}">{PRIORITY_LABELS[priority]} Priority</span>'

        # Appointment details — rendered as pure inline HTML in one block (no floating div issue)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(badge, unsafe_allow_html=True)
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="field-label">Doctor</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="field-value">👨‍⚕️ {appt["doctor_name"]}</p>', unsafe_allow_html=True)
            st.markdown('<p class="field-label">Department</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="field-value">🏥 {appt["department"]}</p>', unsafe_allow_html=True)
        with col2:
            st.markdown('<p class="field-label">Specialty</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="field-value">🩺 {appt["specialty"].title()}</p>', unsafe_allow_html=True)
            st.markdown('<p class="field-label">Date & Time</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="field-value">📅 {appt["slot_datetime"]}</p>', unsafe_allow_html=True)

        st.markdown('<p class="field-label">Patient</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="field-value">✉️ {appt["patient_email"]}</p>', unsafe_allow_html=True)

        st.divider()

        # Medical report
        if "summary" in appt:
            st.markdown("### 📋 AI Medical Report")
            st.caption("⚠️ AI-generated — preliminary only, subject to physician review")
            st.markdown("**Clinical Summary**")
            st.info(appt["summary"])
            if appt.get("medications") and appt["medications"] != "None":
                st.markdown("**Medication Recommendations**")
                st.warning(appt["medications"])
        else:
            st.info("Medical report is being generated — refresh in a moment.")

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Chat", use_container_width=True):
            _go("chat")
    with col2:
        if st.button("Start Over (New Patient)", use_container_width=True):
            for k in ["screen", "email", "executor", "messages", "appointment_id", "booking_done"]:
                if k == "screen":     st.session_state[k] = "email"
                elif k == "messages": st.session_state[k] = []
                elif k == "booking_done": st.session_state[k] = False
                else:                 st.session_state[k] = None
            st.rerun()
