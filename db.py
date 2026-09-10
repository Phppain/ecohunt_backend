from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    nickname = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    camera_permission = Column(Boolean, default=False)
    geo_permission = Column(Boolean, default=False)

    points = Column(Integer, default=0)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    reports = relationship("Report", back_populates="user")

# Friends table
class Friend(Base):
    __tablename__ = "friends"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"))
    friend_id = Column(Integer, ForeignKey("users.id"))

    status = Column(String, default="pending")

class FriendRequest(Base):
    __tablename__ = "friend_requests"

    id = Column(Integer, primary_key=True, index=True)

    sender_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    receiver_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    accepted = Column(
        Boolean,
        default=False
    )

# Reports table
class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)

    lat = Column(Float)
    lng = Column(Float)

    image_before = Column(String)
    image_after = Column(String, nullable=True)

    reports_count = Column(Integer, default=1)
    severity = Column(String, default="green")

    ai_score = Column(Float, nullable=True)
    ai_points_awarded = Column(Integer, nullable=True)
    ai_cleaned = Column(Boolean, nullable=True)

    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="reports")

Base.metadata.create_all(bind=engine)