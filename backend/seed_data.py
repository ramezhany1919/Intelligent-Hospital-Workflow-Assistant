from datetime import datetime, timedelta
from backend.database import engine, get_session
from backend.models import Base, Doctor, DoctorSlot

DOCTORS = [
    {"name": "Dr. Sarah Mitchell",  "specialty": "cardiology",     "department": "Cardiology"},
    {"name": "Dr. James Okafor",    "specialty": "neurology",      "department": "Neurology"},
    {"name": "Dr. Lena Hoffmann",   "specialty": "general",        "department": "General Practice"},
    {"name": "Dr. Ahmed Al-Rashid", "specialty": "orthopedics",    "department": "Orthopedics"},
    {"name": "Dr. Priya Sharma",    "specialty": "dermatology",    "department": "Dermatology"},
    {"name": "Dr. Carlos Mendes",   "specialty": "gastroenterology","department": "Gastroenterology"},
    {"name": "Dr. Fatima Zahra",    "specialty": "pulmonology",    "department": "Pulmonology"},
    {"name": "Dr. Wei Chen",        "specialty": "endocrinology",  "department": "Endocrinology"},
]

SLOTS_PER_DOCTOR = 10
START_DATE = datetime(2026, 4, 25, 9, 0)  # slots start tomorrow morning


def seed():
    Base.metadata.create_all(bind=engine)

    db = get_session()
    try:
        if db.query(Doctor).count() > 0:
            print("Database already seeded. Skipping.")
            return

        for doc_data in DOCTORS:
            doctor = Doctor(**doc_data, is_available=True)
            db.add(doctor)
            db.flush()

            for i in range(SLOTS_PER_DOCTOR):
                slot_time = START_DATE + timedelta(days=i // 2, hours=(i % 2) * 2)
                slot = DoctorSlot(doctor_id=doctor.id, slot_datetime=slot_time, is_booked=False)
                db.add(slot)

        db.commit()
        print(f"Seeded {len(DOCTORS)} doctors with {SLOTS_PER_DOCTOR} slots each.")
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    seed()
