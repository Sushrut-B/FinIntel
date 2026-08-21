import os
import bcrypt
import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Create data directory if not exists
db_dir = Path("data")
db_dir.mkdir(parents=True, exist_ok=True)

DATABASE_URL = "sqlite:///data/finintel.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ----------------------------------------------------
# ORM Models
# ----------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="guest")

class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"
    id = Column(Integer, primary_key=True, index=True)
    ds = Column(String, index=True, nullable=False)
    y = Column(Float, nullable=False)
    forecast = Column(Float, nullable=False)
    residual = Column(Float, nullable=False)
    type = Column(String, nullable=False)

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True, nullable=False)
    version = Column(String, unique=True, nullable=False)
    train_date = Column(String, nullable=False)
    mae = Column(Float, nullable=False)
    mape = Column(Float, nullable=False)
    file_path = Column(String, nullable=False)
    status = Column(String, default="Active")  # Active, Deprecated, Rejected

# Create Tables
Base.metadata.create_all(bind=engine)

# ----------------------------------------------------
# Password Hashing Helpers
# ----------------------------------------------------
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

# ----------------------------------------------------
# Database Seeding on Startup
# ----------------------------------------------------
def seed_database():
    session = SessionLocal()
    try:
        # 1. Seed Users
        existing_sushrut = session.query(User).filter(User.username == "sushrut").first()
        if not existing_sushrut:
            sushrut = User(
                username="sushrut",
                password_hash=hash_password("sushrutpass"),
                role="admin"
            )
            session.add(sushrut)
            print("Seeded database user: sushrut")
            
        existing_admin = session.query(User).filter(User.username == "admin").first()
        if not existing_admin:
            admin = User(
                username="admin",
                password_hash=hash_password("adminpass"),
                role="admin"
            )
            session.add(admin)
            print("Seeded database user: admin")

        # 2. Seed Anomalies from existing CSV if empty
        existing_anomaly_count = session.query(AnomalyRecord).count()
        if existing_anomaly_count == 0:
            csv_path = Path("data/gold/upi_anomalies.csv")
            if csv_path.exists():
                print("Seeding anomalies from existing CSV file...")
                import pandas as pd
                try:
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        record = AnomalyRecord(
                            ds=str(row["ds"]),
                            y=float(row["y"]),
                            forecast=float(row["forecast"]),
                            residual=float(row["residual"]),
                            type=str(row["type"])
                        )
                        session.add(record)
                    print(f"Seeded {len(df)} anomalies records.")
                except Exception as e:
                    print(f"Error seeding anomalies from CSV: {e}")

        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Database seeding failed: {e}")
    finally:
        session.close()

# Run seeding
seed_database()

# ----------------------------------------------------
# DB Session Context Manager
# ----------------------------------------------------
class get_db_session:
    def __enter__(self):
        self.session = SessionLocal()
        return self.session
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
        self.session.close()
