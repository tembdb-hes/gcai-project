##############################################
# Created By Edwin Tembo, Anjay Sukuru, Jayden Chen
##############################################

import contextlib
import aiofiles
import json
import asyncio
import time 
from pathlib import Path
from fastapi import FastAPI, Request , Form, Depends, HTTPException, status, Cookie, Body
from fastapi.security  import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyCookie
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors       import CORSMiddleware
from fastapi.staticfiles           import StaticFiles

from auth import (get_user, 
                  create_user,
                  check_admin,
                  store_access_token, 
                  validate_access_token, 
                  verify_password, 
                  authenticate_user, 
                  create_access_token, 
                  validate_current_user,
                  get_current_user,  
                  password_hash, 
                  delete_access_token)
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse,  JSONResponse, RedirectResponse
from db_models import   UserDB, AccessTokenDB, Token, engine, create_all_tables, async_db_session, end_db_session
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker,  relationship, Session
from sqlalchemy.ext.asyncio   import AsyncSession

from datetime import datetime, timedelta
from typing   import Optional, Annotated
from schemas  import UserCreation


import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import os

import pandas as pd  
import datetime
from inference import ModelInference
import time
import subprocess
import base64

TOKEN_COOKIE_NAME ='access_token'


class liveStreamSetting:
	STREAM = True


@contextlib.asynccontextmanager
async def lifespan(app:FastAPI):
	await create_all_tables()
	yield




app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:8000", 
    "https://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)


home          = os.path.expanduser("~")
app_dir       = os.environ["APP_DIR"]
static_dir    = os.path.join(app_dir, "static")
model_dir     = os.path.join(app_dir, "modelv1")
utils_dir     = os.path.join(app_dir, "utilsv1")
data_dir      = os.path.join(app_dir, "data")
template_dir  = os.path.join(app_dir, "templates")


# Serve static files (HTML)
app.mount("/static", StaticFiles(directory=static_dir),   name="static")
app.mount("/modelv1",  StaticFiles(directory=model_dir),  name="modelv1")
app.mount("/utilsv1",  StaticFiles(directory=utils_dir),  name="utilsv1")
app.mount("/data",   StaticFiles(directory=utils_dir),    name="data")
templates = Jinja2Templates(directory="static")



# Initialize Inference Class
inferenceModel = ModelInference()
DEBUG          = inferenceModel.debug
	
@app.get("/login")
async def login_page(request: Request):

    response = templates.TemplateResponse("login.html", {"request": request})
    
    return response	
    
@app.get("/register", response_class=HTMLResponse)
async def register_page(request     : Request,
                        access_token:str  = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)), 
                        username    :str = Depends(APIKeyCookie(name="username")),  
                        db: AsyncSession = Depends(async_db_session)
                       ):
							
    user = await validate_current_user(access_token,username,db)
    if user.access_level != 'ADMIN':
        raise HTTPException(status_code=401, detail="Unauthorized.")
		
    return templates.TemplateResponse("register.html", {"request": request})
    


@app.post("/register")  
async def register(request: Request, 
                   db                  : AsyncSession = Depends(async_db_session), 
                   access_token        : str          = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)),
                   authorized_username : str          = Depends(APIKeyCookie(name="username")),
                   username            : str = Form(...), 
                   password            : str = Form(...)):
	##validate the admin user
    await validate_current_user(access_token,authorized_username, db)
    ##the new user
    new_user = await get_user(db, username=username)
    if new_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    try:
        user = await create_user(db, username, password)        
        return JSONResponse( status_code=200, content={"detail": "OK"})
    except IntegrityError as e:
        if 'UNIQUE constraint failed' in str(e):
            raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected Error.")

@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(async_db_session)):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.Please register an account if you have not done so.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_expires_min = inferenceModel.access_token_exp_minutes
    access_token, expires_at = create_access_token(
                                                   data={"sub": user.username},
                                                   expires_delta=timedelta(minutes=token_expires_min),
                                                  )
    await store_access_token(db, access_token, expires_at, user)
    if user.access_level != 'ADMIN' or user.access_level is None :		
        response = RedirectResponse(url=f'/', status_code = status.HTTP_303_SEE_OTHER)
    else : 
        response = RedirectResponse(url='/admin', status_code = status.HTTP_303_SEE_OTHER)
        
    response.set_cookie(key=TOKEN_COOKIE_NAME , value=access_token,  httponly=True,max_age=token_expires_min*60 ,secure=True,samesite='lax' )
    response.set_cookie(key="username"        , value=user.username, httponly=True,max_age=token_expires_min*60, secure=True,samesite='lax'  )

    #reset positiond for any new sessions
    inferenceModel.current_pos_x = 0
    inferenceModel.current_pos_y = 0
    inferenceModel.current_scale = 1.0
    return response
    
	
@app.get("/last_transform")
async def last_transform(  access_token        : str          = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)),
                           username            : str          = Depends(APIKeyCookie(name="username")),
                            db                 : AsyncSession = Depends(async_db_session)
                        ):
							   
    user = await validate_current_user(access_token, username , db)   
    transform = { "pos_x"  : inferenceModel.current_pos_x,
                  "pos_y"  : inferenceModel.current_pos_y,
                  "scale"  : inferenceModel.current_scale
                }
                

    response = JSONResponse(status_code=200, content = transform)
    return response
    
@app.post("/logout")
async def end_token_session( access_token : str          = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)),  
                             username     : str          = Depends(APIKeyCookie(name="username")),
                            db            : AsyncSession = Depends(async_db_session)):
    user = await validate_current_user(access_token, username , db)
    if user:
        resp = await delete_access_token(access_token, db)
        
    response = RedirectResponse(url=f'/login', status_code = status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key=TOKEN_COOKIE_NAME )
    response.delete_cookie(key="username")
    return response
        
    

def gen_plain(): 
	cam = cv.VideoCapture(inferenceModel.cap_device)
	while True:
		success, frame = cam.read()
		if not success:
			if DEGUG:
			    print("no success")
		
		else:
			frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			ret, buffer = cv.imencode('.jpg', frame)
			frame = buffer.tobytes()
			
			yield (b'--frame\r\n'
			       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
	cam.release()
	cv.destroyAllWindows()
			      
@app.get("/plain_video")
def pain_video_feed(access_token : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)), 
                    username     : str = Depends(APIKeyCookie(name="username")), 
                    db           : AsyncSession = Depends(async_db_session) ):
	user = asyncio.run(validate_current_user(access_token, username, db))					
	return StreamingResponse(gen_plain(), media_type = "multipart/x-mixed-replace; boundary=frame")
	
	
@app.post("/color_space")
def change_color_space(selected_space  : str ,
                       access_token : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)), 
                       username     : str = Depends(APIKeyCookie(name="username")), 
                       db           : AsyncSession = Depends(async_db_session) ):
						   
	user = asyncio.run(validate_current_user(access_token, username, db))

	if selected_space == 'XYZ':
		inferenceModel.color_convert = cv.COLOR_BGR2XYZ
	elif selected_space == 'LUV' :
		inferenceModel.color_convert = cv.COLOR_BGR2LUV		
	else :
		inferenceModel.color_convert = cv.COLOR_BGR2RGB



def restart_cam():
	try:
		cv.destroyAllWindows()
		time.sleep(0.5)
	except Exception as e:
		print(f"cam destroy error {e}")
	print('inference device')
	print(inferenceModel.cap_device)
	
	
	
def video_inference(multipart_boundary = None, status=None): 

	restart_cam()
	try:
		cam = cv.VideoCapture(inferenceModel.cap_device)
	except Exception as e:
		print(f"Exception : {e}")
		
	restart_cam()
	    
	while liveStreamSetting.STREAM:		
		success, frame = cam.read()
		counter = 0
		
		if not success :
			 print("no success")
			 
			 liveStreamSetting.STREAM = False

		
		else:
			
			if inferenceModel.color_convert is not None and inferenceModel.color_convert != '' :
				image = cv.cvtColor(frame, inferenceModel.color_convert) 
			else:
				image = frame

			###run inference
			image = cv.flip(image, 1)
			debug_image = copy.deepcopy(image)
			frame,data = inferenceModel.run_inference(image, debug_image )
			yield ('--boundary={multipart_boundary}\r\n'
			       'Content-Type: application/json\r\n\r\n'	+ json.dumps(data) + '\r\n\r\n'
			       '--boundary={multipart_boundary}\r\n'			       
			       'Content-Type: image/jpeg\r\n\r\n' + frame + '\r\n')

		       
	cam.release()
	cv.destroyAllWindows()
		


	

def video_inference_multi(multipart_boundary = None, status=None): 

	restart_cam()
	try:
		cam = cv.VideoCapture(inferenceModel.cap_device)
		cam2 = cv.VideoCapture(inferenceModel.cap_device2)
	except Exception as e:
		print(f"Exception : {e}")
		
	restart_cam()
	    
	while liveStreamSetting.STREAM:		
		success, frame = cam.read()
		success2, frame2 = cam2.read()
		counter = 0
		
		if not success :
			 print("no success")
			 
			 liveStreamSetting.STREAM = False

		
		else:
			
			if inferenceModel.color_convert is not None and inferenceModel.color_convert != '' :
				image = cv.cvtColor(frame, inferenceModel.color_convert) 
			else:
				image = frame
			
			##live stream camera ##	
			image2 = frame2
			_, buffer2 = cv.imencode('.jpg', image2)
			frame2 = base64.b64encode(buffer2).decode('utf-8')	
				       
			###run inference
			image = cv.flip(image, 1)
			debug_image = copy.deepcopy(image)
			frame,data = inferenceModel.run_inference(image, debug_image )

			yield ('--boundary={multipart_boundary}\r\n'
			       'Content-Type: application/json\r\n\r\n'	+ json.dumps(data) + '\r\n\r\n'
			       '--boundary={multipart_boundary}\r\n'			       
			       'Content-Type: image/jpeg\r\n\r\n' + frame  + '\r\n'
			       '--boundary={multipart_boundary}\r\n'
			       'Content-Type: image/jpeg\r\n\r\n' + frame2 + '\r\n'
			       )

		       
	cam.release()
	cv.destroyAllWindows()
	

			      
@app.get("/video")
def video_feed(access_token : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)), 
               username     : str = Depends(APIKeyCookie(name="username")), 
               db           : AsyncSession = Depends(async_db_session),
               status       : str = None):
	
	user = asyncio.run(validate_current_user(access_token, username, db))
	liveStreamSetting.STREAM = True
	boundary = 'nmahjsousojVzvbag'
	return StreamingResponse(video_inference(boundary,status), media_type=f"multipart/x-mixed-replace; boundary={boundary}")
	
	
	
@app.get("/video_multi")
def video_feed(access_token : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)), 
               username     : str = Depends(APIKeyCookie(name="username")), 
               db           : AsyncSession = Depends(async_db_session),
               status       : str = None):
	
	user = asyncio.run(validate_current_user(access_token, username, db))
	liveStreamSetting.STREAM = True
	boundary = 'nmahjsousojVzvbag'
	return StreamingResponse(video_inference_multi(boundary,status), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


			      
@app.post("/stop_stream")
async def stop_stream( clear_state  : str = 'N',
                      access_token  : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)) , 
                      username      : str = Depends(APIKeyCookie(name="username")),
                      db            : AsyncSession = Depends(async_db_session)):  
    user = await validate_current_user(access_token,username, db)
    liveStreamSetting.STREAM = False

		


@app.get("/datastream")
async def render_ds_page(access_token  : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)) , 
                         username      : str = Depends(APIKeyCookie(name="username")),
                         db            : AsyncSession = Depends(async_db_session) ):
    user = await validate_current_user(access_token,username, db)    
    async with aiofiles.open(r"static/datastream.html", mode="r") as f:
        return HTMLResponse(await f.read())



@app.get("/")
async def main(access_token  : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)),  
               username      : str = Depends(APIKeyCookie(name="username")),
               db            : AsyncSession = Depends(async_db_session)):
     
   user = await validate_current_user(access_token, username, db)
   async with aiofiles.open(r"static/home.html", mode="r") as f:
       return HTMLResponse(await f.read())
       

@app.get("/imaging_live_stream")
async def main(access_token  : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)),  
               username      : str = Depends(APIKeyCookie(name="username")),
               db            : AsyncSession = Depends(async_db_session)):
     
   user = await validate_current_user(access_token, username, db)
   async with aiofiles.open(r"static/home_video.html", mode="r") as f:
       return HTMLResponse(await f.read())


      
@app.get("/unauth", response_class=HTMLResponse)       
async def unauth_user():
	async with aiofiles.open(r"static/home.html", mode="r") as f:
		return HTMLResponse(await f.read())
	
	       
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request       : Request,
                          access_token  : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)) ,  
                          username      : str = Depends(APIKeyCookie(name="username")),
                          db            : AsyncSession = Depends(async_db_session) ):
  
    user = await validate_current_user(access_token, username, db)
    if user.access_level != 'ADMIN':
        raise HTTPException(status_code=401, detail="Unauthorized.")
   
    return templates.TemplateResponse("admin.html", {"request": request})      
       
       
      
# Store image position and zoom state
image_state = {
    "imgX": 0,
    "imgY": 0,
    "zoom": 1
}


# API endpoint to get image state
@app.get("/image-state")
def get_image_state(access_token  : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)) , 
                    username      : str = Depends(APIKeyCookie(name="username")),
                    db            : AsyncSession = Depends(async_db_session)
                    ):
						
    user = asyncio.run(validate_current_user(access_token, username, db))					
						
    return image_state

# API endpoint to update image state
@app.post("/update-image-state")
def update_image_state(x     =  Body(...),    
                       y     =  Body(...),
                       zoom  =  Body(...) ,          
                       access_token  : str = Depends(APIKeyCookie(name=TOKEN_COOKIE_NAME)) , 
                       username      : str = Depends(APIKeyCookie(name="username")),
                       db            : AsyncSession = Depends(async_db_session)
                       ):
						   
			   				   
    asyncio.run(validate_current_user(access_token, username, db))	
    global image_state
    image_state["imgX"] = x
    image_state["imgY"] = y
    image_state["zoom"] = zoom
    return image_state


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
