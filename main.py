from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from db import SessionLocal, User, Friend, Report
from schemas import *
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

import cv2
from skimage.metrics import structural_similarity as ssim

from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Или ["http://localhost:5173", "http://localhost:8000"] для более безопасного варианта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _save_upload(file: UploadFile, prefix: str) -> str:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"]:
        # keep it simple; clients can still upload jpg/png
        ext = ".jpg"
    name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path

def analyze_cleanup(before_path: str, after_path: str):
    """
    Free local "AI" (no paid APIs): compares before/after using SSIM + edge density.
    Returns (ai_score 0..1, cleaned bool, points_awarded int).
    """
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)
    if before is None or after is None:
        return None, None, 0

    before_g = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_g = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    after_g = cv2.resize(after_g, (before_g.shape[1], before_g.shape[0]))

    score = float(ssim(before_g, after_g))

    # Simple "cleanup" signal: fewer edges after cleaning.
    before_edges = cv2.Canny(before_g, 80, 160)
    after_edges = cv2.Canny(after_g, 80, 160)
    before_density = float(before_edges.mean())
    after_density = float(after_edges.mean())

    cleaned = (score < 0.92) and (after_density < before_density * 0.95)

    # Points: reward meaningful change + cleaner edge density.
    delta = max(0.0, (before_density - after_density))
    points = int(min(50, max(5, delta * 300)))
    if not cleaned:
        points = int(max(0, points // 4))

    # Normalize an "AI score" where higher means "better cleaned"
    ai_score = max(0.0, min(1.0, (before_density - after_density) * 8))
    return ai_score, cleaned, points

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    return user

# ---------------- AUTH ----------------
@app.post("/auth/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(nickname=user.nickname, email=user.email, hashed_password=pwd_context.hash(user.password))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/auth/login", response_model=Token)
def login(user: UserOut, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = jwt.encode(
        {"user_id": db_user.id, "exp": datetime.utcnow() + timedelta(days=7)},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserOut)
def get_me(user: User = Depends(get_current_user)):
    return user

# ---------------- USERS ----------------
@app.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    return user

@app.put("/users/me", response_model=UserOut)
def update_profile(data: UserCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user.nickname = data.nickname
    user.email = data.email
    db.commit()
    db.refresh(user)
    return user

@app.patch("/users/me/permissions", response_model=UserOut)
def update_permissions(
    camera: bool = False,
    geolocation: bool = False,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user.camera_permission = camera
    user.geo_permission = geolocation
    db.commit()
    db.refresh(user)
    return user

# ---------------- FRIENDS ----------------
@app.post("/friends/add", response_model=FriendOut)
def add_friend(friend: FriendCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    friend_user = db.query(User).filter(User.nickname == friend.nickname).first()

    if not friend_user:
        raise HTTPException(status_code=404, detail="User not found")

    exists = db.query(Friend).filter(
        Friend.user_id == user.id,
        Friend.friend_id == friend_user.id
    ).first()

    if exists:
        raise HTTPException(status_code=400, detail="Already friends")

    db_friend = Friend(user_id=user.id, friend_id=friend_user.id)
    db.add(db_friend)
    db.commit()
    db.refresh(db_friend)

    return friend_user

@app.get("/friends", response_model=List[FriendOut])
def get_friends(db: Session = Depends(get_db)):
    friends = db.query(User).all()
    return friends

import random

@app.get("/friends/locations")
def friends_locations(db: Session = Depends(get_db)):
    friends = db.query(User).all()

    return [
        {
            "id": f.id,
            "nickname": f.nickname,
            "lat": 43.2 + random.uniform(-0.05, 0.05),
            "lng": 76.9 + random.uniform(-0.05, 0.05)
        }
        for f in friends
    ]
# ---------------- REPORTS ----------------
@app.post("/reports", response_model=ReportOut)
def create_report(
    lat: float = Form(...),
    lng: float = Form(...),
    image_before: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    before_path = _save_upload(image_before, "before")

    db_report = Report(
        lat=lat,
        lng=lng,
        image_before=before_path,
        user_id=user.id
    )

    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    return db_report

@app.post("/reports/{report_id}/clean", response_model=ReportOut)
def clean_report(
    report_id: int,
    image_after: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    after_path = _save_upload(image_after, "after")
    report.image_after = after_path

    # Free local analysis
    ai_score, cleaned, points_awarded = analyze_cleanup(report.image_before, report.image_after)
    report.ai_score = ai_score
    report.ai_cleaned = cleaned
    report.ai_points_awarded = points_awarded

    # Apply points to user (simple demo: first user)
    user = user
    if user and points_awarded:
        user.points = (user.points or 0) + points_awarded

    report.reports_count += 1
    report.severity = "red" if report.reports_count >= 8 else "yellow"
    db.commit()
    db.refresh(report)
    return report

@app.get("/reports", response_model=List[ReportOut])
def get_reports(db: Session = Depends(get_db)):
    return db.query(Report).all()


# ---------------- LEADERBOARD ----------------
@app.get("/leaderboard/global", response_model=List[LeaderboardEntry])
def global_leaderboard(db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.points.desc()).all()
    return [{"nickname": u.nickname, "points": u.points} for u in users]

@app.get("/leaderboard/friends", response_model=List[LeaderboardEntry])
def friends_leaderboard(db: Session = Depends(get_db)):
    # простой пример, все пользователи
    users = db.query(User).all()
    return [{"nickname": u.nickname, "points": u.points} for u in users]