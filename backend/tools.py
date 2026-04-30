import smtplib
import os
import json
import pandas as pd
from email.mime.text import MIMEText
from langchain_core.tools import tool
from backend.database import get_session
from backend.models import User, Doctor, DoctorSlot, Appointment, MedicalReport


def _parse_str(raw, key):
    """Extract a single value when the ReAct agent passes a JSON string as a single-param tool arg."""
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            return json.loads(raw)[key]
        except Exception:
            pass
    return raw


def _parse_json(raw):
    """Parse a JSON string or passthrough a dict — used for multi-param tools."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


# ── Single-param tools (no Pydantic conflict) ─────────────────────────────────

@tool
def get_patient_history(email: str) -> dict:
    """
    Retrieve the patient's profile and past appointment history from the database.
    Call this at the start of every intake session before asking any follow-up questions.
    Returns age, chronic disease history, and a list of previous visits.
    Input: email (string)
    """
    email = _parse_str(email, "email")
    db = get_session()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return {"success": False, "error": f"No user found with email {email}"}

        past_appointments = []
        for appt in user.appointments:
            entry = {
                "appointment_id": appt.id,
                "doctor": appt.doctor.name,
                "specialty": appt.doctor.specialty,
                "slot": appt.slot.slot_datetime.strftime("%Y-%m-%d %H:%M"),
                "priority": appt.priority,
            }
            if appt.medical_report:
                entry["report_summary"] = appt.medical_report.summary
            past_appointments.append(entry)

        return {
            "success": True,
            "data": {
                "email": user.email,
                "age": user.age,
                "chronic_disease_history": user.chronic_disease_history,
                "past_appointments": past_appointments,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        db.close()


@tool
def find_available_doctor(specialty: str) -> dict:
    """
    Find an available doctor for the given medical specialty and return their next open time slot.
    Specialty must be one of: cardiology, neurology, general, orthopedics, dermatology,
    gastroenterology, pulmonology, endocrinology.
    Use 'general' if no specific specialty is required.
    Input: specialty (string)
    """
    specialty = _parse_str(specialty, "specialty")
    db = get_session()
    try:
        doctor = (
            db.query(Doctor)
            .filter(Doctor.specialty == specialty.lower(), Doctor.is_available == True)
            .first()
        )
        if not doctor:
            return {"success": False, "error": f"No available doctor found for specialty: {specialty}"}

        slot = (
            db.query(DoctorSlot)
            .filter(DoctorSlot.doctor_id == doctor.id, DoctorSlot.is_booked == False)
            .order_by(DoctorSlot.slot_datetime)
            .first()
        )
        if not slot:
            return {"success": False, "error": f"Dr. {doctor.name} has no available slots"}

        return {
            "success": True,
            "data": {
                "doctor_id": doctor.id,
                "doctor_name": doctor.name,
                "specialty": doctor.specialty,
                "department": doctor.department,
                "slot_id": slot.id,
                "slot_datetime": slot.slot_datetime.strftime("%Y-%m-%d %H:%M"),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        db.close()


# ── Multi-param tools: accept a single JSON string to avoid Pydantic conflict ──

@tool
def book_appointment(input_json: str) -> dict:
    """
    Book an appointment by linking the patient to a doctor's time slot.
    Input must be a JSON string with keys:
      user_email (str), doctor_id (int), slot_id (int), priority (int).
    Priority: 0=urgent (chest pain, stroke), 1=high, 2=moderate, 3=routine.
    This atomically marks the slot as booked so no two patients can claim the same slot.
    Example: {"user_email": "alice@example.com", "doctor_id": 1, "slot_id": 3, "priority": 1}
    """
    try:
        params = _parse_json(input_json)
        user_email = params["user_email"]
        doctor_id  = int(params["doctor_id"])
        slot_id    = int(params["slot_id"])
        priority   = int(params["priority"])
    except Exception as e:
        return {"success": False, "error": f"Invalid input — expected JSON with user_email, doctor_id, slot_id, priority. Error: {e}"}

    if priority not in (0, 1, 2, 3):
        return {"success": False, "error": "Priority must be 0 (urgent), 1, 2, or 3 (routine)"}

    db = get_session()
    try:
        slot = db.query(DoctorSlot).filter(DoctorSlot.id == slot_id).with_for_update().first()
        if not slot:
            return {"success": False, "error": f"Slot {slot_id} not found"}
        if slot.is_booked:
            return {"success": False, "error": f"Slot {slot_id} was just taken. Please call find_available_doctor again."}

        slot.is_booked = True
        appointment = Appointment(
            user_email=user_email,
            doctor_id=doctor_id,
            slot_id=slot_id,
            priority=priority,
        )
        db.add(appointment)
        db.commit()
        db.refresh(appointment)

        doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
        return {
            "success": True,
            "data": {
                "appointment_id": appointment.id,
                "doctor_name": doctor.name,
                "specialty": doctor.specialty,
                "slot_datetime": slot.slot_datetime.strftime("%Y-%m-%d %H:%M"),
                "priority": priority,
            },
        }
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    finally:
        db.close()


@tool
def save_medical_report(input_json: str) -> dict:
    """
    Save the AI-generated clinical summary and medication recommendations for an appointment.
    Always call this after booking the appointment.
    Input must be a JSON string with keys:
      appointment_id (int), summary (str), medication_recommendations (str, optional).
    The summary should describe the patient's symptoms, triage assessment, and reasoning.
    Example: {"appointment_id": 5, "summary": "...", "medication_recommendations": "..."}
    """
    try:
        params = _parse_json(input_json)
        appointment_id         = int(params["appointment_id"])
        summary                = params["summary"]
        medication_recommendations = params.get("medication_recommendations", "None")
    except Exception as e:
        return {"success": False, "error": f"Invalid input — expected JSON with appointment_id, summary. Error: {e}"}

    db = get_session()
    try:
        appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
        if not appt:
            return {"success": False, "error": f"Appointment {appointment_id} not found"}

        report = MedicalReport(
            appointment_id=appointment_id,
            summary=summary,
            medication_recommendations=medication_recommendations or "None",
        )
        db.add(report)
        db.commit()
        return {"success": True, "data": {"report_id": report.id, "appointment_id": appointment_id}}
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    finally:
        db.close()


@tool
def send_confirmation_email(input_json: str) -> dict:
    """
    Send an appointment confirmation email to the patient.
    Call this as the last step after saving the medical report.
    Input must be a JSON string with keys:
      to_email (str), doctor_name (str), slot_datetime (str), department (str).
    If SMTP is not configured, the confirmation will be logged to the console instead.
    Example: {"to_email": "alice@example.com", "doctor_name": "Dr. Smith", "slot_datetime": "2026-04-25 09:00", "department": "Cardiology"}
    """
    try:
        params        = _parse_json(input_json)
        to_email      = params["to_email"]
        doctor_name   = params["doctor_name"]
        slot_datetime = params["slot_datetime"]
        department    = params["department"]
    except Exception as e:
        return {"success": False, "error": f"Invalid input — expected JSON with to_email, doctor_name, slot_datetime, department. Error: {e}"}

    subject = "Your Appointment Confirmation — Medica"
    body = (
        f"Dear Patient,\n\n"
        f"Your appointment has been successfully booked.\n\n"
        f"  Doctor    : {doctor_name}\n"
        f"  Department: {department}\n"
        f"  Date/Time : {slot_datetime}\n\n"
        f"Please note: This booking was assisted by an AI system. "
        f"All recommendations are preliminary and subject to physician review.\n\n"
        f"See you soon,\nMedica"
    )

    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")

    if smtp_host and smtp_user and smtp_pass:
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = to_email
            with smtplib.SMTP_SSL(smtp_host, 465) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, to_email, msg.as_string())
            return {"success": True, "data": {"sent_to": to_email, "method": "smtp"}}
        except Exception as e:
            return {"success": False, "error": f"SMTP failed: {str(e)}"}
    else:
        print(f"\n[EMAIL LOG] To: {to_email}\nSubject: {subject}\n{body}\n")
        return {"success": True, "data": {"sent_to": to_email, "method": "logged"}}


@tool
def get_doctor_slots(doctor_id: str) -> dict:
    """
    Return all time slots for a given doctor — both available and booked.
    Useful for inspecting a doctor's full schedule.
    Input: doctor_id as a string (e.g. "1")
    """
    try:
        did = int(_parse_str(doctor_id, "doctor_id"))
    except (ValueError, TypeError):
        return {"success": False, "error": f"Invalid doctor_id: {doctor_id}"}

    db = get_session()
    try:
        doctor = db.query(Doctor).filter(Doctor.id == did).first()
        if not doctor:
            return {"success": False, "error": f"No doctor found with id {did}"}

        slots = (
            db.query(DoctorSlot)
            .filter(DoctorSlot.doctor_id == did)
            .order_by(DoctorSlot.slot_datetime)
            .all()
        )
        return {
            "success": True,
            "data": {
                "doctor_id": doctor.id,
                "doctor_name": doctor.name,
                "specialty": doctor.specialty,
                "slots": [
                    {
                        "slot_id": s.id,
                        "datetime": s.slot_datetime.strftime("%Y-%m-%d %H:%M"),
                        "status": "booked" if s.is_booked else "available",
                    }
                    for s in slots
                ],
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        db.close()


@tool
def get_all_appointments(_: str = "") -> dict:
    """
    Return all appointments in the system with patient, doctor, slot, and priority details.
    Use this to inspect the current state of bookings.
    No input required — pass an empty string.
    """
    db = get_session()
    try:
        appointments = db.query(Appointment).order_by(Appointment.id).all()
        return {
            "success": True,
            "data": [
                {
                    "appointment_id": a.id,
                    "patient_email": a.user_email,
                    "doctor": a.doctor.name,
                    "specialty": a.doctor.specialty,
                    "slot_datetime": a.slot.slot_datetime.strftime("%Y-%m-%d %H:%M"),
                    "priority": a.priority,
                    "has_report": a.medical_report is not None,
                }
                for a in appointments
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        db.close()


@tool
def get_medical_report(appointment_id: str) -> dict:
    """
    Retrieve the medical report for a given appointment, if it exists.
    Input: appointment_id as a string (e.g. "5")
    """
    try:
        aid = int(_parse_str(appointment_id, "appointment_id"))
    except (ValueError, TypeError):
        return {"success": False, "error": f"Invalid appointment_id: {appointment_id}"}

    db = get_session()
    try:
        report = (
            db.query(MedicalReport)
            .filter(MedicalReport.appointment_id == aid)
            .first()
        )
        if not report:
            return {"success": False, "error": f"No medical report found for appointment {aid}"}

        return {
            "success": True,
            "data": {
                "report_id": report.id,
                "appointment_id": report.appointment_id,
                "summary": report.summary,
                "medication_recommendations": report.medication_recommendations,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        db.close()


# ── Notebook utility (not an agent tool) ──────────────────────────────────────

def show_appointments(email_filter: str = "") -> pd.DataFrame:
    """
    Return a formatted DataFrame of all appointments in the DB.
    Optionally filter by patient email (partial match, case-insensitive).

    Usage in notebook:
        from backend.tools import show_appointments
        show_appointments()                        # all appointments
        show_appointments("alice@example.com")     # specific patient
        show_appointments("example.com")           # partial match
    """
    PRIORITY_LABELS = {0: "P0 — Urgent", 1: "P1 — High", 2: "P2 — Moderate", 3: "P3 — Routine"}
    db = get_session()
    try:
        q = db.query(Appointment)
        if email_filter.strip():
            q = q.filter(Appointment.user_email.ilike(f"%{email_filter.strip()}%"))
        rows = []
        for a in q.order_by(Appointment.id).all():
            rows.append({
                "ID":          a.id,
                "Patient":     a.user_email,
                "Doctor":      a.doctor.name,
                "Specialty":   a.doctor.specialty.title(),
                "Date & Time": a.slot.slot_datetime.strftime("%Y-%m-%d %H:%M"),
                "Priority":    PRIORITY_LABELS.get(a.priority, str(a.priority)),
                "Report":      "Yes" if a.medical_report else "No",
            })
    finally:
        db.close()

    if not rows:
        msg = "No appointments found" + (f" for '{email_filter}'" if email_filter else "") + "."
        print(msg)
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("ID")