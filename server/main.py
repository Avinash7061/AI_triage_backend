"""
Healthcare Triage System — FastAPI Backend (Production v3.0)
Full hospital panel with real-time rush, patient records, MySQL storage.
"""

import os
import uuid
import time
import math
from datetime import datetime, timedelta, timezone

import jwt
import bcrypt
import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session, joinedload

from server.database import get_db, init_db
from server.models import User, Hospital, Department, PatientRecord, Prescription, Appointment

# ─── Configuration ───────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", "mediflow-ai-secret-key-2024-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

HF_SPACE_URL = os.environ.get(
    "HF_SPACE_URL",
    "https://avi7061-mediflow-triage.hf.space"
)

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

LABEL_DESC = {
    "Red":    "🔴 Emergency — Immediate attention required",
    "Orange": "🟠 Urgent — See a doctor within 1-2 days",
    "Yellow": "🟡 Semi-urgent — Schedule within 1 week",
    "White":  "⚪ Home care — Rest and self-care recommended",
}

# ─── FastAPI App ─────────────────────────────────────────────
app = FastAPI(title="MediFlow AI — Triage API", version="3.0.0")

allowed_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "https://lifecure.org.in",
    "https://www.lifecure.org.in",
    "https://ai-triage-app-psi.vercel.app",
    FRONTEND_URL,
]
if os.environ.get("RAILWAY_PUBLIC_DOMAIN"):
    allowed_origins.append(f"https://{os.environ['RAILWAY_PUBLIC_DOMAIN']}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print(f"\n{'='*60}")
    print(f"  MediFlow AI — Backend Server v3.0 (Hospital Panel)")
    print(f"{'='*60}")
    print(f"  HF Space URL  : {HF_SPACE_URL}")
    print(f"  Frontend URL   : {FRONTEND_URL}")
    init_db()
    print(f"{'='*60}\n")

# ─── JWT Helpers ─────────────────────────────────────────────

def create_token(user_dict: dict) -> str:
    payload = {
        "sub": user_dict["id"],
        "email": user_dict["email"],
        "name": user_dict["name"],
        "role": user_dict["role"],
        "hospitalId": user_dict.get("hospitalId"),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ")[1]
    return decode_token(token)

# ─── Request Models ─────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    role: str
    hospitalName: Optional[str] = None
    hospitalLocation: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class PrescriptionCreate(BaseModel):
    patientId: str
    patientName: str
    symptoms: list
    aiSuggestion: str
    triageCategory: str

class PrescriptionVerify(BaseModel):
    notes: str
    status: str

class AppointmentCreate(BaseModel):
    hospitalId: str
    departmentName: str
    slot: str

class HospitalUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    emergencyAvailable: Optional[bool] = None

class DepartmentCreate(BaseModel):
    name: str
    capacity: int = 50

class RushUpdate(BaseModel):
    change: int  # +1 or -1

class PatientRecordCreate(BaseModel):
    patientName: str
    age: Optional[int] = None
    gender: Optional[str] = None
    symptoms: Optional[list] = None
    triageCategory: Optional[str] = None
    roomNumber: Optional[str] = None
    departmentId: Optional[str] = None
    notes: Optional[str] = None

class PatientRecordUpdate(BaseModel):
    status: Optional[str] = None
    roomNumber: Optional[str] = None
    notes: Optional[str] = None
    departmentId: Optional[str] = None
    triageCategory: Optional[str] = None

# ─── Prediction Endpoint ─────────────────────────────────────

@app.post("/api/predict")
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{HF_SPACE_URL}/predict",
                json={"text": req.text},
            )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Model API error: {response.text}")
        result = response.json()
        total_time = (time.time() - start_time) * 1000
        return {
            "prediction": result["prediction"],
            "description": LABEL_DESC.get(result["prediction"], result.get("description", "")),
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "inference_time_ms": round(total_time, 1),
        }
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to model API. HuggingFace Space may be sleeping.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Model API timed out. HuggingFace Space may be cold-starting.")

# ─── Auth Endpoints ──────────────────────────────────────────

@app.post("/api/auth/register")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if req.role not in ("patient", "doctor", "hospital_staff"):
        raise HTTPException(status_code=400, detail="Invalid role")

    existing = db.query(User).filter(User.email == req.email.lower()).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

    # For hospital_staff, require hospital details
    hospital_id = None
    if req.role == "hospital_staff":
        if not req.hospitalName or not req.hospitalLocation:
            raise HTTPException(status_code=400, detail="Hospital name and location are required for hospital staff")
        hospital = Hospital(
            id=str(uuid.uuid4()),
            name=req.hospitalName,
            location=req.hospitalLocation,
            owner_id=str(uuid.uuid4()),  # Will update after user creation
            emergency_available=False,
        )
        db.add(hospital)
        db.flush()
        hospital_id = hospital.id

    hashed = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user = User(
        id=str(uuid.uuid4()),
        email=req.email.lower(),
        name=req.name,
        role=req.role,
        password_hash=hashed,
        hospital_id=hospital_id,
    )
    db.add(user)

    # Update hospital owner_id to the actual user id
    if req.role == "hospital_staff" and hospital_id:
        hospital.owner_id = user.id

    db.commit()
    db.refresh(user)

    token = create_token(user.to_dict())
    return {"token": token, "user": user.to_dict()}

@app.post("/api/auth/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email.lower()).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not bcrypt.checkpw(req.password.encode("utf-8"), user.password_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user.to_dict())
    return {"token": token, "user": user.to_dict()}

@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "user": {
            "id": current_user["sub"],
            "email": current_user["email"],
            "name": current_user["name"],
            "role": current_user["role"],
            "hospitalId": current_user.get("hospitalId"),
        }
    }

# ─── Hospital Endpoints ──────────────────────────────────────

@app.get("/api/hospitals")
async def list_hospitals(db: Session = Depends(get_db)):
    """Public endpoint — list all hospitals with departments for patients."""
    hospitals = db.query(Hospital).options(joinedload(Hospital.departments)).all()
    return [h.to_dict(include_departments=True) for h in hospitals]

@app.get("/api/hospitals/mine")
async def get_my_hospital(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the hospital belonging to the logged-in staff member."""
    if current_user["role"] != "hospital_staff":
        raise HTTPException(status_code=403, detail="Only hospital staff can access this")
    hospital_id = current_user.get("hospitalId")
    if not hospital_id:
        raise HTTPException(status_code=404, detail="No hospital linked to this account")
    hospital = db.query(Hospital).options(joinedload(Hospital.departments)).filter(Hospital.id == hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
    return hospital.to_dict(include_departments=True)

@app.put("/api/hospitals/mine")
async def update_my_hospital(
    req: HospitalUpdate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] != "hospital_staff":
        raise HTTPException(status_code=403, detail="Only hospital staff can access this")
    hospital_id = current_user.get("hospitalId")
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
    if req.name is not None:
        hospital.name = req.name
    if req.location is not None:
        hospital.location = req.location
    if req.emergencyAvailable is not None:
        hospital.emergency_available = req.emergencyAvailable
    db.commit()
    db.refresh(hospital)
    return hospital.to_dict()

# ─── Department Endpoints ─────────────────────────────────────

@app.post("/api/hospitals/mine/departments")
async def add_department(
    req: DepartmentCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] != "hospital_staff":
        raise HTTPException(status_code=403, detail="Only hospital staff can access this")
    hospital_id = current_user.get("hospitalId")
    if not hospital_id:
        raise HTTPException(status_code=404, detail="No hospital linked")
    dept = Department(
        id=str(uuid.uuid4()),
        hospital_id=hospital_id,
        name=req.name,
        capacity=req.capacity,
        current_patients=0,
        avg_wait_time=0,
    )
    db.add(dept)
    db.commit()
    db.refresh(dept)
    return dept.to_dict()

@app.patch("/api/departments/{dept_id}/rush")
async def update_rush(
    dept_id: str,
    req: RushUpdate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] != "hospital_staff":
        raise HTTPException(status_code=403, detail="Only hospital staff can access this")
    dept = db.query(Department).filter(Department.id == dept_id).first()
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    # Verify ownership
    if dept.hospital_id != current_user.get("hospitalId"):
        raise HTTPException(status_code=403, detail="Not your department")
    dept.current_patients = max(0, dept.current_patients + req.change)
    # Auto-calculate wait time based on load ratio
    if dept.capacity > 0:
        load_ratio = dept.current_patients / dept.capacity
        dept.avg_wait_time = math.ceil(load_ratio * 45)  # ~45 min at full capacity
    db.commit()
    db.refresh(dept)
    return dept.to_dict()

@app.delete("/api/departments/{dept_id}")
async def delete_department(
    dept_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] != "hospital_staff":
        raise HTTPException(status_code=403, detail="Only hospital staff can access this")
    dept = db.query(Department).filter(Department.id == dept_id).first()
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    if dept.hospital_id != current_user.get("hospitalId"):
        raise HTTPException(status_code=403, detail="Not your department")
    db.delete(dept)
    db.commit()
    return {"message": "Department deleted"}

# ─── Patient Records Endpoints ────────────────────────────────

@app.get("/api/patient-records")
async def list_patient_records(
    status: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] != "hospital_staff":
        raise HTTPException(status_code=403, detail="Only hospital staff can access this")
    hospital_id = current_user.get("hospitalId")
    query = db.query(PatientRecord).filter(PatientRecord.hospital_id == hospital_id)
    if status:
        query = query.filter(PatientRecord.status == status)
    records = query.order_by(PatientRecord.admitted_at.desc()).all()
    return [r.to_dict() for r in records]

@app.post("/api/patient-records")
async def admit_patient(
    req: PatientRecordCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] != "hospital_staff":
        raise HTTPException(status_code=403, detail="Only hospital staff can access this")
    hospital_id = current_user.get("hospitalId")
    if not hospital_id:
        raise HTTPException(status_code=404, detail="No hospital linked")
    record = PatientRecord(
        id=str(uuid.uuid4()),
        hospital_id=hospital_id,
        department_id=req.departmentId,
        patient_name=req.patientName,
        age=req.age,
        gender=req.gender,
        symptoms=req.symptoms,
        triage_category=req.triageCategory,
        room_number=req.roomNumber,
        notes=req.notes,
        status="admitted",
    )
    db.add(record)
    # Auto-increment department patient count
    if req.departmentId:
        dept = db.query(Department).filter(Department.id == req.departmentId).first()
        if dept:
            dept.current_patients = max(0, dept.current_patients + 1)
            if dept.capacity > 0:
                dept.avg_wait_time = math.ceil((dept.current_patients / dept.capacity) * 45)
    db.commit()
    db.refresh(record)
    return record.to_dict()

@app.patch("/api/patient-records/{record_id}")
async def update_patient_record(
    record_id: str,
    req: PatientRecordUpdate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] != "hospital_staff":
        raise HTTPException(status_code=403, detail="Only hospital staff can access this")
    record = db.query(PatientRecord).filter(PatientRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Patient record not found")
    if record.hospital_id != current_user.get("hospitalId"):
        raise HTTPException(status_code=403, detail="Not your hospital's record")

    if req.status is not None:
        old_status = record.status
        record.status = req.status
        # Auto-decrement department count on discharge
        if req.status in ("discharged", "transferred") and old_status == "admitted" and record.department_id:
            dept = db.query(Department).filter(Department.id == record.department_id).first()
            if dept:
                dept.current_patients = max(0, dept.current_patients - 1)
                if dept.capacity > 0:
                    dept.avg_wait_time = math.ceil((dept.current_patients / dept.capacity) * 45)
        if req.status == "discharged":
            record.discharged_at = datetime.now(timezone.utc)
    if req.roomNumber is not None:
        record.room_number = req.roomNumber
    if req.notes is not None:
        record.notes = req.notes
    if req.departmentId is not None:
        record.department_id = req.departmentId
    if req.triageCategory is not None:
        record.triage_category = req.triageCategory

    db.commit()
    db.refresh(record)
    return record.to_dict()

# ─── Prescriptions ───────────────────────────────────────────

@app.get("/api/prescriptions")
async def list_prescriptions(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Prescription)
    if current_user["role"] == "patient":
        query = query.filter(Prescription.patient_id == current_user["sub"])
    rxs = query.order_by(Prescription.created_at.desc()).all()
    return [rx.to_dict() for rx in rxs]

@app.post("/api/prescriptions")
async def create_prescription(
    req: PrescriptionCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rx = Prescription(
        id=str(uuid.uuid4()),
        patient_id=req.patientId,
        patient_name=req.patientName,
        symptoms=req.symptoms,
        ai_suggestion=req.aiSuggestion,
        triage_category=req.triageCategory,
    )
    db.add(rx)
    db.commit()
    db.refresh(rx)
    return rx.to_dict()

@app.patch("/api/prescriptions/{rx_id}/verify")
async def verify_prescription(
    rx_id: str,
    req: PrescriptionVerify,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can verify prescriptions")
    rx = db.query(Prescription).filter(Prescription.id == rx_id).first()
    if not rx:
        raise HTTPException(status_code=404, detail="Prescription not found")
    rx.verified = True
    rx.status = req.status
    rx.doctor_notes = req.notes
    db.commit()
    db.refresh(rx)
    return rx.to_dict()

# ─── Appointments ─────────────────────────────────────────────

@app.get("/api/appointments")
async def list_appointments(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Appointment)
    if current_user["role"] == "patient":
        query = query.filter(Appointment.patient_id == current_user["sub"])
    appts = query.order_by(Appointment.created_at.desc()).all()
    return [a.to_dict() for a in appts]

@app.post("/api/appointments")
async def create_appointment(
    req: AppointmentCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    appt = Appointment(
        id=str(uuid.uuid4()),
        patient_id=current_user["sub"],
        hospital_id=req.hospitalId,
        department_name=req.departmentName,
        slot=req.slot,
    )
    db.add(appt)
    db.commit()
    db.refresh(appt)
    return appt.to_dict()

# ─── Health Check ────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.0.0", "hf_space": HF_SPACE_URL}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

