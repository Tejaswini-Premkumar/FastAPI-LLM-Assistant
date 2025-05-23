from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


DATABASE_URL = "sqlite:///./tickets.db"

# SQLAlchemy engine

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create a declarative base for our models
Base = declarative_base()

# Define the Ticket model (table)
class Ticket(Base):
    __tablename__ = "tickets" # Name of the table in the database

    id = Column(Integer, primary_key=True, index=True) # Unique ID for each ticket
    subject = Column(String, index=True) # Subject of the ticket
    description = Column(Text) # Detailed description, using Text for potentially long strings
    product_name = Column(String, default="N/A") # Product related to the ticket
    user_id = Column(String, default="N/A") # User who raised the ticket
    priority = Column(String, default="Medium") # Priority (High, Medium, Low)
    extracted_status = Column(String, default="success") # Status of extraction
    created_at = Column(DateTime, default=datetime.utcnow) # Timestamp of creation

    def __repr__(self):
        # String representation for debugging
        return f"<Ticket(id={self.id}, subject='{self.subject}', priority='{self.priority}')>"


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_db_tables():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")


def get_db():
    db = SessionLocal()
    try:
        yield db # Yields the session, which will be closed after the request
    finally:
        db.close() # Ensures the session is closed