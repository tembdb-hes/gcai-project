from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Cookie
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from db_models import UserDB, AccessTokenDB,  engine, async_db_session
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import relationship
from sqlalchemy import  create_engine, select , delete
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import Optional, Annotated
from pwdlib import PasswordHash
from collections.abc import AsyncGenerator
import secrets 
import os



"""
Manage Authentication and User Security 
"""


# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

password_hash = PasswordHash.recommended()


async def get_user(db: AsyncSession , username: str):
	query = select(UserDB).where(UserDB.username == username)
	result = await db.execute(query)
	user : UserDB  | None = result.scalar_one_or_none()
	return user
	

async def create_user(db: AsyncSession, username : str , password : str):

    hashed_password = password_hash.hash(password)
    db_user = UserDB(username=username, hashed_password=hashed_password, access_level="GENERAL")
    db.add(db_user)
    await db.commit()
    db.refresh(db_user)
    return db_user

async def store_access_token(db: AsyncSession, token: str, expires_at: datetime, user: UserDB):
    db_token = AccessTokenDB(token=token, expires_at=expires_at, user=user)
    db.add(db_token)
    await db.commit()
    return db_token
    
async def check_admin(db : AsyncSession , password : str = None):
    query = select(AccessTokenDB).where(AccessTokenDB.username == 'admin'
                                       ).limit(1)
                                                                      
    result =  await db.execute(query)
    admin_user : AccessTokenDB  | None = result.scalar_one_or_none()
                           
    if not admin_user or admin_user is None:
        admin_user = await create_user(db, 'admin', 'admin' if password is None else password)
        print("\n*************************Admin Created**********************************************\n")
    return admin_user
	
	
	
	   
    
async def validate_access_token(token : str , db: AsyncSession):
    query = select(AccessTokenDB).where(AccessTokenDB.token == token, 
                                        AccessTokenDB.expires_at > datetime.utcnow()
                                       ).limit(1)
                                                                      
    result =  await db.execute(query)
    db_token : AccessTokenDB  | None = result.scalar_one_or_none()
                           
    if not db_token or db_token is None or db_token.expires_at < datetime.utcnow():
        return None
    return db_token

async def delete_access_token(token : str,  db: AsyncSession):
     query    =  delete(AccessTokenDB).where(AccessTokenDB.token== token )
     result   =  await db.execute(query)    
     await db.commit()
                                                  
	    

# Authentication helper functions
async def verify_password(plain_password, hashed_password):
   
    return password_hash.verify(plain_password, hashed_password) 

async def authenticate_user(db: AsyncSession, username: str, password: str):
    user = await get_user(db, username)
    password_verif = None
    if user is not None:
        password_verif = await verify_password(password, user.hashed_password)
    if not user or not  password_verif:
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=1440))
    to_encode.update({"exp": expire})
    
    token = secrets.token_urlsafe(512)
    return token, expire
    


async def get_current_user(access_token : str, username :str,  db : AsyncSession):

    db_token= await validate_access_token(access_token, db)
                                                                                               
    if db_token is None : 
        
        raise HTTPException(status_code=401, detail="Invalid or expired token.Please Log In.")
    if not db_token.user.username == username:
        raise HTTPException(status_code=401, detail="Invalid or expired token.Please Log In.")
    return db_token.user
 
    return None
    
async def validate_current_user(access_token : str , username :str, db : AsyncSession):
 
    user = None
    if access_token:
        access_token = access_token.replace("bearer ","").strip()
        user = await get_current_user(access_token ,username, db)
    if not user:
        response = RedirectResponse(url=f'/unauth', status_code = status.HTTP_303_SEE_OTHER)
        return response
        raise HTTPException(status_code=401, detail="Session invalid or expired. Please login.")
    return user
