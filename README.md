# Medica — Intelligent Hospital Workflow Assistant
**Deep Learning (DLCV) — Milestone 2**

An AI-powered patient intake system that automates triage, doctor booking, medical report generation, and email confirmation through a LangChain ReAct agent backed by Claude claude-sonnet-4-6.

---

## System Architecture

```
Patient (Browser)
      │
      ▼
Streamlit Frontend  ──►  FastAPI Backend  ──►  LangChain ReAct Agent (Claude claude-sonnet-4-6)
   frontend/app.py        backend/main.py           backend/agent.py
                                │                         │
                                ▼                         ▼
                         PostgreSQL DB              Tools (7 @tool functions)
                         backend/models.py          backend/tools.py
```

### Agent Workflow (8 steps, every session)
1. `get_patient_history` — retrieve profile + past visits
2. Ask ONE focused follow-up question if symptoms are ambiguous
3. Assign priority (P0–P3) and identify medical specialty
4. `find_available_doctor` — find next open slot
5. `book_appointment` — atomic slot reservation (no double-booking)
6. `save_medical_report` — persist clinical summary + medication notes
7. `send_confirmation_email` — notify patient via SMTP or console log
8. Present appointment details to patient

### Priority Levels
| Level | Label | Trigger examples |
|---|---|---|
| P0 | Urgent | Chest pain, stroke symptoms, loss of consciousness |
| P1 | High | Fever >39°C, severe pain, sudden vision/hearing loss |
| P2 | Moderate | Persistent pain, worsening chronic condition, infection |
| P3 | Routine | Checkups, mild symptoms >1 week, prescription renewals |

---

## Project Structure

```
DLCV-Milestone 2/
├── Milestone_2.ipynb       # Main deliverable — full demo notebook
├── requirements.txt
├── .env                    # API keys + DB URL (not committed)
├── backend/
│   ├── database.py         # SQLAlchemy engine + session factory
│   ├── models.py           # ORM: User, Doctor, DoctorSlot, Appointment,
│   │                       #       MedicalReport, ConversationHistory
│   ├── seed_data.py        # seed() and reset_and_reseed(n) helpers
│   ├── tools.py            # 7 LangChain @tool functions + show_appointments()
│   ├── agent.py            # ReAct agent builder (build_agent_executor)
│   └── main.py             # FastAPI: /identify /register /chat /appointment/{id}
├── frontend/
│   └── app.py              # Streamlit UI (4 screens + agent reasoning trace)
└── Agent Screenshots/      # UI screenshots and demo recording
```

---

## Setup

### 1. Prerequisites
- Python 3.10+
- PostgreSQL running locally

### 2. Create the database
```bash
createdb hospital_workflow
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install langchain-classic streamlit
```

### 4. Configure environment
Create a `.env` file in the project root:
```env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://localhost/hospital_workflow

# Optional — only needed for real confirmation emails
SMTP_HOST=smtp.gmail.com
SMTP_USER=your@gmail.com
SMTP_PASS=your_app_password
```

### 5. Seed the database
```bash
python3 -m backend.seed_data
```
This inserts 8 doctors with 60 slots each. To reset and reseed at any time:
```python
from backend.seed_data import reset_and_reseed
reset_and_reseed(60)
```

---

## Running

### Notebook (primary deliverable)
Open `Milestone_2.ipynb` in Jupyter and run all cells top to bottom.

### Streamlit frontend
```bash
streamlit run frontend/app.py
```

### FastAPI backend (standalone)
```bash
uvicorn backend.main:app --reload
```
Endpoints: `POST /identify` · `POST /register` · `POST /chat` · `GET /appointment/{id}`

---

## Doctors & Specialties

| Doctor | Specialty | Department |
|---|---|---|
| Dr. Sarah Mitchell | cardiology | Cardiology |
| Dr. James Okafor | neurology | Neurology |
| Dr. Lena Hoffmann | general | General Practice |
| Dr. Ahmed Al-Rashid | orthopedics | Orthopedics |
| Dr. Priya Sharma | dermatology | Dermatology |
| Dr. Carlos Mendes | gastroenterology | Gastroenterology |
| Dr. Fatima Zahra | pulmonology | Pulmonology |
| Dr. Wei Chen | endocrinology | Endocrinology |

---

## Evaluation Results

Priority triage accuracy evaluated across 7 labelled test cases:

| # | Symptom | Expected | Predicted | Result |
|---|---------|----------|-----------|--------|
| 1 | Severe chest pain radiating to left arm | P0 | N/A* | FAIL |
| 2 | Sudden loss of speech and facial drooping | P0 | P0 | PASS |
| 3 | High fever 39.5°C + severe headache (2 days) | P1 | P1 | PASS |
| 4 | Moderate knee pain after a fall, ambulatory | P2 | P2 | PASS |
| 5 | Dry cough and mild fatigue for 1 week | P2 | P3 | FAIL |
| 6 | Routine blood pressure check, no symptoms | P3 | P3 | PASS |
| 7 | Prescription renewal for metformin | P3 | P3 | PASS |

**Accuracy: 71% (5/7)**

*Case 1 failure is a system constraint (unregistered test patient → FK violation), not a model reasoning error — the agent correctly identified P0 and attempted to book immediately.

---

## Key Design Decisions

- **Single `input_json: str` pattern** for multi-param tools — avoids Pydantic validation conflicts with `langchain_classic`'s raw string passing behavior.
- **Atomic slot booking** — `SELECT ... FOR UPDATE` prevents two patients claiming the same slot concurrently.
- **Baseline snapshot** — the frontend captures the max appointment ID at chat start so it only triggers the booking-complete screen for appointments created in the current session.
- **No diagnosis** — all agent responses explicitly state assessments are preliminary and subject to physician review.

---

## Notes

- The agent reasoning trace (Thought/Action/Observation loop) is visible in the Streamlit UI via an expandable "Agent reasoning trace" panel.
- Confirmation emails are sent via Gmail SMTP if credentials are configured; otherwise the email body is printed to the console.
- `show_appointments(email_filter="")` in `backend/tools.py` is a plain utility function (not an agent tool) for inspecting the DB from the notebook.
