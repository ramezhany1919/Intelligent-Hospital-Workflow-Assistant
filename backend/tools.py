import smtplib
import os
from email.mime.text import MIMEText
from typing import Optional
from langchain_core.tools import tool
from backend.database import get_session
from backend.models import User, Doctor, DoctorSlot, Appointment, MedicalReport


@tool
def get_patient_history(email: str) -> dict:
    """
    Retrieve the patient's profile and past appointment history from the database.
    Call this at the start of every intake session before asking any follow-up questions.
    Returns age, chronic disease history, and a list of previous visits.
    """
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
    """
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


@tool
def book_appointment(user_email: str, doctor_id: int, slot_id: int, priority: int) -> dict:
    """
    Book an appointment by linking the patient to a doctor's time slot.
    Priority: 0=urgent (chest pain, stroke), 1=high, 2=moderate, 3=routine.
    This atomically marks the slot as booked so no two patients can claim the same slot.
    """
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
def save_medical_report(appointment_id: int, summary: str, medication_recommendations: Optional[str] = "None") -> dict:
    """
    Save the AI-generated clinical summary and medication recommendations for an appointment.
    Always call this after booking the appointment.
    The summary should describe the patient's symptoms, triage assessment, and reasoning.
    """
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
def send_confirmation_email(to_email: str, doctor_name: str, slot_datetime: str, department: str) -> dict:
    """
    Send an appointment confirmation email to the patient.
    Call this as the last step after saving the medical report.
    If SMTP is not configured, the confirmation will be logged instead.
    """
    subject = "Your Appointment Confirmation — Hospital Workflow Assistant"
    body = (
        f"Dear Patient,\n\n"
        f"Your appointment has been successfully booked.\n\n"
        f"  Doctor    : {doctor_name}\n"
        f"  Department: {department}\n"
        f"  Date/Time : {slot_datetime}\n\n"
        f"Please note: This booking was assisted by an AI system. "
        f"All recommendations are preliminary and subject to physician review.\n\n"
        f"See you soon,\nHospital Workflow Assistant"
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
