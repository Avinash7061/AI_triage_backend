"""
SQLAlchemy ORM Models — Users, Hospitals, Departments, PatientRecords, Prescriptions, Appointments
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, Boolean, DateTime, JSON, ForeignKey, Float
from sqlalchemy.orm import relationship
from server.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)  # patient, doctor, hospital_staff
    password_hash = Column(String(255), nullable=False)
    hospital_id = Column(String(36), ForeignKey("hospitals.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "hospitalId": self.hospital_id,
        }


class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=False)
    owner_id = Column(String(36), nullable=False, index=True)
    emergency_available = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    departments = relationship("Department", back_populates="hospital", cascade="all, delete-orphan")

    def to_dict(self, include_departments=True):
        data = {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "ownerId": self.owner_id,
            "emergencyAvailable": self.emergency_available,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }
        if include_departments and self.departments:
            data["departments"] = [d.to_dict() for d in self.departments]
        else:
            data["departments"] = []
        return data


class Department(Base):
    __tablename__ = "departments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    hospital_id = Column(String(36), ForeignKey("hospitals.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    capacity = Column(Integer, default=50)
    current_patients = Column(Integer, default=0)
    avg_wait_time = Column(Integer, default=0)  # minutes
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    hospital = relationship("Hospital", back_populates="departments")

    def to_dict(self):
        return {
            "id": self.id,
            "hospitalId": self.hospital_id,
            "name": self.name,
            "capacity": self.capacity,
            "currentPatients": self.current_patients,
            "averageWaitTime": self.avg_wait_time,
        }


class PatientRecord(Base):
    __tablename__ = "patient_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    hospital_id = Column(String(36), ForeignKey("hospitals.id"), nullable=False, index=True)
    department_id = Column(String(36), ForeignKey("departments.id"), nullable=True)
    patient_name = Column(String(255), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    symptoms = Column(JSON, nullable=True)
    triage_category = Column(String(20), nullable=True)  # Red, Orange, Yellow, White
    room_number = Column(String(50), nullable=True)
    status = Column(String(20), default="admitted", index=True)  # admitted, discharged, transferred
    notes = Column(Text, nullable=True)
    admitted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    discharged_at = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "hospitalId": self.hospital_id,
            "departmentId": self.department_id,
            "patientName": self.patient_name,
            "age": self.age,
            "gender": self.gender,
            "symptoms": self.symptoms or [],
            "triageCategory": self.triage_category,
            "roomNumber": self.room_number,
            "status": self.status,
            "notes": self.notes,
            "admittedAt": self.admitted_at.isoformat() if self.admitted_at else None,
            "dischargedAt": self.discharged_at.isoformat() if self.discharged_at else None,
        }


class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    patient_id = Column(String(36), nullable=False, index=True)
    patient_name = Column(String(255), nullable=False)
    symptoms = Column(JSON, nullable=False)
    ai_suggestion = Column(Text, nullable=False)
    triage_category = Column(String(20), nullable=False)
    verified = Column(Boolean, default=False)
    status = Column(String(20), default="pending")
    doctor_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "patientId": self.patient_id,
            "patientName": self.patient_name,
            "symptoms": self.symptoms or [],
            "aiSuggestion": self.ai_suggestion,
            "triageCategory": self.triage_category,
            "verified": self.verified,
            "status": self.status,
            "doctorNotes": self.doctor_notes,
            "timestamp": self.created_at.isoformat() if self.created_at else None,
        }


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    patient_id = Column(String(36), nullable=False, index=True)
    hospital_id = Column(String(50), nullable=False)
    department_name = Column(String(255), nullable=False)
    slot = Column(String(100), nullable=False)
    status = Column(String(20), default="booked")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "patientId": self.patient_id,
            "hospitalId": self.hospital_id,
            "departmentName": self.department_name,
            "slot": self.slot,
            "status": self.status,
        }
