
from collections.abc import AsyncGenerator
from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, DateTime, create_engine
from sqlalchemy.orm import sessionmaker,  relationship, Session, mapped_column, Mapped, DeclarativeBase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from pydantic import BaseModel
from typing import Optional

"""
Database schema models for user management
"""

class Base(DeclarativeBase):
	pass
	



# Database Models
class UserDB(Base):
    __tablename__ = "users"
    id :Mapped[int] = mapped_column(primary_key=True, index=True,autoincrement=True )
    username : Mapped[str]  = mapped_column(String(1024), unique=True, nullable=False, index=True)
    hashed_password : Mapped[str]= mapped_column(String(1024))
    access_level    : Mapped[str]= mapped_column(String(1024) , default = "GENERAL")
    

class AccessTokenDB(Base):
    __tablename__ = "access_tokens"
    id         = Column(Integer, primary_key=True, index=True)
    token      = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    user_id    = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped[UserDB]  = relationship("UserDB", lazy="joined")

class Token(BaseModel):
    access_token: str
    token_type: str
    
    
    
class ColorSpace(Base):
    __tablename__ = "color_space"
    id                    = Column(Integer, primary_key=True, index=True)
    space_name            = Column(String, default = 'RGB')
    user_id               = mapped_column(ForeignKey("users.id"), unique=True, nullable=False)
    user: Mapped[UserDB]  = relationship("UserDB", lazy="joined")
    
    
# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./users.db"
engine = create_async_engine(SQLALCHEMY_DATABASE_URL)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

       
async def async_db_session() -> AsyncGenerator[AsyncSession, None]:
	async with async_session_maker() as session:
		yield session
		
async def end_db_session(session: AsyncSession) -> AsyncGenerator[AsyncSession, None]:
	session.close()
	

	
	

async def create_all_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all) 
        



        
