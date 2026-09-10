from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from db import SessionLocal, User, Friend, Report, FriendRequest
from schemas import *
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import numpy as np
import gc
from ultralytics import YOLO

import cv2

from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"

model = YOLO("yolov8n.pt")

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

MAX_SIZE = 800

def load_image(path: str):
    img = cv2.imread(path)

    if img is None:
        return None

    return img

def detect_trash(image_path):

    results = model.predict(
        image_path,
        imgsz=640,
        conf=0.35,
        verbose=False
    )

    count = 0

    for result in results:
        count += len(result.boxes)

    return count

def _save_upload(file: UploadFile, prefix: str) -> str:
    name = f"{prefix}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, name)

    data = file.file.read()

    img = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise HTTPException(400, "Invalid image")

    h, w = img.shape[:2]

    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)

        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    cv2.imwrite(
        path,
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    del img
    gc.collect()

    return path

def analyze_cleanup(before_path, after_path):

    before_count = detect_trash(before_path)
    after_count = detect_trash(after_path)

    removed = max(
        0,
        before_count - after_count
    )

    cleaned = removed > 0

    if before_count == 0:
        ai_score = 1.0
    else:
        ai_score = removed / before_count

    points = removed * 10

    return (
        ai_score,
        cleaned,
        points
    )

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    return user

# ---------------- AUTH ----------------
@app.post("/auth/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.email == user.email).first()

    if exists:
        raise HTTPException(status_code=400, detail="Email already exists")

    db_user = User(
        nickname=user.nickname,
        email=user.email,
        hashed_password=pwd_context.hash(user.password[:72]),
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    token = jwt.encode(
        {
            "user_id": db_user.id,
            "exp": datetime.utcnow() + timedelta(days=7),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    return {
        "access_token": token,
        "token_type": "bearer",
    }

@app.post("/auth/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
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
    print(user)
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

@app.patch("/users/location")
def update_location(
    data: UpdateLocation,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user.latitude = data.lat
    user.longitude = data.lng

    db.commit()
    db.refresh(user)

    return {
        "lat": user.latitude,
        "lng": user.longitude
    }

# ---------------- FRIENDS ----------------
@app.post("/friends/request")
def send_friend_request(
    friend: FriendCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    friend_user = db.query(User).filter(
        User.nickname == friend.nickname
    ).first()


    if not friend_user:
        raise HTTPException(
            404,
            "User not found"
        )


    if friend_user.id == user.id:
        raise HTTPException(
            400,
            "Can't add yourself"
        )


    exists = db.query(FriendRequest).filter(
        FriendRequest.sender_id == user.id,
        FriendRequest.receiver_id == friend_user.id,
        FriendRequest.accepted == False
    ).first()


    if exists:
        raise HTTPException(
            400,
            "Request already exists"
        )


    request = FriendRequest(
        sender_id=user.id,
        receiver_id=friend_user.id
    )


    db.add(request)
    db.commit()
    db.refresh(request)


    return {
        "request_id": request.id
    }

@app.get("/friends/requests")
def get_friend_requests(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    requests = db.query(FriendRequest).filter(
        FriendRequest.receiver_id == user.id,
        FriendRequest.accepted == False
    ).all()


    result = []

    for r in requests:

        sender = db.query(User).filter(
            User.id == r.sender_id
        ).first()


        result.append({
            "request_id": r.id,
            "id": sender.id,
            "nickname": sender.nickname,
            "email": sender.email,
            "points": sender.points
        })


    return result

@app.post("/friends/requests/{request_id}/accept")
def accept_request(
    request_id:int,
    user:User = Depends(get_current_user),
    db:Session = Depends(get_db)
):

    request = db.query(FriendRequest).filter(
        FriendRequest.id == request_id,
        FriendRequest.receiver_id == user.id
    ).first()


    if not request:
        raise HTTPException(
            404,
            "Request not found"
        )


    db.add(
        Friend(
            user_id=request.sender_id,
            friend_id=request.receiver_id
        )
    )


    db.add(
        Friend(
            user_id=request.receiver_id,
            friend_id=request.sender_id
        )
    )


    request.accepted = True

    db.commit()


    return {
        "message":"Friend added"
    }

@app.delete("/friends/requests/{request_id}")
def decline_request(
    request_id:int,
    user:User = Depends(get_current_user),
    db:Session = Depends(get_db)
):

    request = db.query(FriendRequest).filter(
        FriendRequest.id == request_id,
        FriendRequest.receiver_id == user.id
    ).first()


    if not request:
        raise HTTPException(
            404,
            "Request not found"
        )


    db.delete(request)
    db.commit()


    return {
        "message":"Declined"
    }

@app.get("/friends", response_model=List[FriendOut])
def get_friends(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    relations = db.query(Friend).filter(
        Friend.user_id == user.id
    ).all()

    ids = [f.friend_id for f in relations]

    if not ids:
        return []

    friends = db.query(User).filter(
        User.id.in_(ids)
    ).all()

    return friends

import random

@app.get("/friends/locations")
def friends_locations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    relations = db.query(Friend).filter(
        Friend.user_id == user.id
    ).all()

    ids = [f.friend_id for f in relations]

    friends = db.query(User).filter(
        User.id.in_(ids)
    ).all()

    return [
    {
        "id": f.id,
        "nickname": f.nickname,
        "lat": f.latitude,
        "lng": f.longitude,
        "points": f.points
    }
    for f in friends
    if f.latitude is not None and f.longitude is not None
]

@app.delete("/friends/{friend_id}")
def remove_friend(
    friend_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.query(Friend).filter(
        Friend.user_id == user.id,
        Friend.friend_id == friend_id
    ).delete()

    db.query(Friend).filter(
        Friend.user_id == friend_id,
        Friend.friend_id == user.id
    ).delete()

    db.commit()

    return {"message": "Friend removed"}
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
    print("Points awarded:", points_awarded)
    print("User points before:", user.points)


    if user and points_awarded:
        user.points = (user.points or 0) + points_awarded

    print("User points after:", user.points)

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
    users = (
        db.query(User)
        .order_by(User.points.desc())
        .all()
    )

    return [
    {
        "rank": i + 1,
        "nickname": u.nickname,
        "points": u.points
    }
    for i, u in enumerate(users)
]

@app.get("/leaderboard/friends", response_model=List[LeaderboardEntry])
def friends_leaderboard(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    relations = db.query(Friend).filter(
        Friend.user_id == user.id
    ).all()

    ids = [f.friend_id for f in relations]
    ids.append(user.id)

    users = (
        db.query(User)
        .filter(User.id.in_(ids))
        .order_by(User.points.desc())
        .all()
    )

    return [
        {
            "nickname": u.nickname,
            "points": u.points
        }
        for u in users
    ]