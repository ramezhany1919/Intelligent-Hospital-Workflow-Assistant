from datetime import datetime, timedelta
from backend.database import engine, get_session
from backend.models import Base, Doctor, DoctorSlot, Appointment, MedicalReport, ConversationHistory, User

DOCTORS = [
    {"name": "Dr. Sarah Mitchell",  "specialty": "cardiology",      "department": "Cardiology"},
    {"name": "Dr. James Okafor",    "specialty": "neurology",       "department": "Neurology"},
    {"name": "Dr. Lena Hoffmann",   "specialty": "general",         "department": "General Practice"},
    {"name": "Dr. Ahmed Al-Rashid", "specialty": "orthopedics",     "department": "Orthopedics"},
    {"name": "Dr. Priya Sharma",    "specialty": "dermatology",     "department": "Dermatology"},
    {"name": "Dr. Carlos Mendes",   "specialty": "gastroenterology","department": "Gastroenterology"},
    {"name": "Dr. Fatima Zahra",    "specialty": "pulmonology",     "department": "Pulmonology"},
    {"name": "Dr. Wei Chen",        "specialty": "endocrinology",   "department": "Endocrinology"},
]

SLOTS_PER_DOCTOR = 10
START_DATE = datetime(2026, 4, 25, 9, 0)


def _insert_doctors(db, slots_per_doctor: int):
    """Insert all doctors and their time slots. Each doctor gets 4 slots per day (9,11,13,15)."""
    hours = [9, 11, 13, 15]
    for doc_data in DOCTORS:
        doctor = Doctor(**doc_data, is_available=True)
        db.add(doctor)
        db.flush()

        slot_count = 0
        day = 0
        while slot_count < slots_per_doctor:
            for hour in hours:
                if slot_count >= slots_per_doctor:
                    break
                slot_time = START_DATE + timedelta(days=day, hours=hour - 9)
                db.add(DoctorSlot(doctor_id=doctor.id, slot_datetime=slot_time, is_booked=False))
                slot_count += 1
            day += 1


def seed():
    """Initial seed — skips if doctors already exist."""
    Base.metadata.create_all(bind=engine)
    db = get_session()
    try:
        if db.query(Doctor).count() > 0:
            print("Database already seeded. Skipping.")
            return
        _insert_doctors(db, SLOTS_PER_DOCTOR)
        db.commit()
        print(f"Seeded {len(DOCTORS)} doctors with {SLOTS_PER_DOCTOR} slots each.")
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def reset_and_reseed(slots_per_doctor: int = 60):
    """
    Drop all data from every table and reseed doctors with the given number of slots.
    Patient records (users), appointments, reports, and conversation history are all cleared.
    """
    Base.metadata.create_all(bind=engine)
    db = get_session()
    try:
        # Delete in FK-safe order
        db.query(MedicalReport).delete()
        db.query(ConversationHistory).delete()
        db.query(Appointment).delete()
        db.query(DoctorSlot).delete()
        db.query(Doctor).delete()
        db.query(User).delete()
        db.commit()
        print("All tables cleared.")

        _insert_doctors(db, slots_per_doctor)
        db.commit()
        print(f"Reseeded {len(DOCTORS)} doctors with {slots_per_doctor} slots each.")
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    reset_and_reseed(60)
