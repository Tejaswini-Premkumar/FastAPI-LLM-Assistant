from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Define the SQLite database URL
# This will create a file named 'tickets.db' in your project directory
DATABASE_URL = "sqlite:///./tickets.db"

# Create the SQLAlchemy engine
# connect_args={"check_same_thread": False} is needed for SQLite with FastAPI
# because SQLite operates on a single thread by default, and FastAPI uses multiple threads.
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

# Create a sessionmaker to produce new Session objects
# Each Session object is a "staging zone" for objects loaded from the database.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to create all tables defined in Base
def create_db_tables():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

# Dependency for FastAPI to get a database session
# This function will be called for each request that needs a database session.
def get_db():
    db = SessionLocal()
    try:
        yield db # Yields the session, which will be closed after the request
    finally:
        db.close() # Ensures the session is closed