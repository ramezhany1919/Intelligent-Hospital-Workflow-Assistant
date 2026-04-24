import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langchain.memory import ConversationBufferMemory

from backend.database import get_db, engine
from backend.models import Base, User, Appointment, ConversationHistory
from backend.agent import build_agent_executor

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Hospital Workflow Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: session_id -> AgentExecutor
_sessions: dict = {}


# ── Request / Response schemas ────────────────────────────────────────────────

class IdentifyRequest(BaseModel):
    email: str

class RegisterRequest(BaseModel):
    email: str
    age: int
    chronic_disease_history: Optional[str] = "None"

class ChatRequest(BaseModel):
    email: str
    message: str
    session_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/identify")
def identify_user(req: IdentifyRequest, db: Session = Depends(get_db)):
    """Check whether a user exists. Returns profile if found, or signals new registration needed."""
    user = db.query(User).filter(User.email == req.email).first()
    if user:
        return {
            "exists": True,
            "email": user.email,
            "age": user.age,
            "chronic_disease_history": user.chronic_disease_history,
        }
    return {"exists": False, "email": req.email}


@app.post("/register")
def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new patient profile."""
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already registered")

    user = User(
        email=req.email,
        age=req.age,
        chronic_disease_history=req.chronic_disease_history,
    )
    db.add(user)
    db.commit()
    return {"success": True, "email": user.email}


@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Send a message to the AI agent. Creates or resumes a session.
    The agent runs the full ReAct loop and returns its response.
    """
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in _sessions:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="input",
        )
        _sessions[session_id] = build_agent_executor(memory)

    executor = _sessions[session_id]

    input_text = f"Patient email: {req.email}\n\nPatient message: {req.message}"

    try:
        result = executor.invoke({"input": input_text})
        response_text = result.get("output", "I'm sorry, I encountered an issue. Please try again.")
    except Exception as e:
        response_text = f"An error occurred: {str(e)}"

    db.add(ConversationHistory(session_id=session_id, user_email=req.email, role="user", content=req.message))
    db.add(ConversationHistory(session_id=session_id, user_email=req.email, role="assistant", content=response_text))
    db.commit()

    return {"session_id": session_id, "response": response_text}


@app.get("/appointment/{appointment_id}")
def get_appointment(appointment_id: int, db: Session = Depends(get_db)):
    """Retrieve confirmed appointment details including doctor and slot info."""
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    result = {
        "appointment_id": appt.id,
        "patient_email": appt.user_email,
        "doctor": appt.doctor.name,
        "specialty": appt.doctor.specialty,
        "department": appt.doctor.department,
        "slot_datetime": appt.slot.slot_datetime.strftime("%Y-%m-%d %H:%M"),
        "priority": appt.priority,
    }
    if appt.medical_report:
        result["medical_summary"] = appt.medical_report.summary
        result["medication_recommendations"] = appt.medical_report.medication_recommendations

    return result


@app.get("/health")
def health():
    return {"status": "ok"}
