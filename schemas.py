from pydantic import BaseModel, EmailStr
from typing import Optional

# ---------------- USERS ----------------

class UpdateLocation(BaseModel):
    lat: float
    lng: float


class UserCreate(BaseModel):
    nickname: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    nickname: str
    email: str

    camera_permission: bool
    geo_permission: bool

    points: int

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


# ---------------- FRIENDS ----------------

class FriendCreate(BaseModel):
    nickname: str


class FriendOut(BaseModel):
    id: int
    nickname: str
    points: int

    class Config:
        orm_mode = True


class FriendLocation(BaseModel):
    id: int
    nickname: str

    lat: float
    lng: float

    points: int


class FriendRequestOut(BaseModel):
    request_id:int
    id:int
    nickname:str
    email:str
    points:int

    class Config:
        orm_mode=True


# ---------------- REPORTS ----------------

class ReportCreate(BaseModel):
    lat: float
    lng: float
    image_before: str


class ReportClean(BaseModel):
    image_after: str


class ReportOut(BaseModel):
    id: int

    lat: float
    lng: float

    severity: str
    reports_count: int

    ai_score: Optional[float] = None
    ai_points_awarded: Optional[int] = None
    ai_cleaned: Optional[bool] = None

    user_id: int

    class Config:
        orm_mode = True


# ---------------- LEADERBOARD ----------------

class LeaderboardEntry(BaseModel):
    nickname: str
    points: int