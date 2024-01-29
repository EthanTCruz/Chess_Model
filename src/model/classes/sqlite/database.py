from sqlalchemy import create_engine

from Chess_Model.src.model.classes.sqlite.models import Base

from sqlalchemy.orm import sessionmaker, Session



DATABASE_URL = "sqlite:///Chess_Model/src/model/data/gameData.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base.metadata.create_all(bind=engine)

