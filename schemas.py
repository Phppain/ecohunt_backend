from pydantic import BaseModel, EmailStr
from typing import Optional, List

# Auth
class UserCreate(BaseModel):
    nickname: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    nickname: str
    email: str
    camera_permission: bool
    geo_permission: bool
    points: int
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

# Friends
class FriendCreate(BaseModel):
    nickname: str

class FriendOut(BaseModel):
    id: int
    nickname: str
    class Config:
        orm_mode = True

# Reports
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
    class Config:
        orm_mode = True

# Leaderboard
class LeaderboardEntry(BaseModel):
    nickname: str
    points: int