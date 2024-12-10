from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles
import json
import asyncio
import time 

from pathlib import Path


######## For MediaPipe #################

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



app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


home = os.path.expanduser("~")
app_dir    = os.environ["APP_DIR"]
static_dir = os.path.join(app_dir, "static")
model_dir  = os.path.join(app_dir, "modelv1")
utils_dir  = os.path.join(app_dir, "utilsv1")
data_dir   = os.path.join(app_dir, "data")

# Serve static files (HTML)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/modelv1",  StaticFiles(directory=model_dir),  name="modelv1")
app.mount("/utilsv1",  StaticFiles(directory=utils_dir),  name="utilsv1")
app.mount("/data",   StaticFiles(directory=utils_dir),  name="data")

# Initialize Inference Class
inferenceModel = ModelInference()
print("AFTER INIT MODEL PARAMS AND CAM")



def gen_plain(): 
	cam = cv.VideoCapture(0)
	while True:
		success, frame = cam.read()
		if not success:
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
def pain_video_feed():
	return StreamingResponse(gen_plain(), media_type = "multipart/x-mixed-replace; boundary=frame")
		

def video_inference(): 
	cam = cv.VideoCapture(0)
	while True:
		success, frame = cam.read()
		if not success:
			print("no success")
		
		else:
			image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

			###run inference
			image = cv.flip(image, 1)
			debug_image = copy.deepcopy(image)
			data = inferenceModel.run_inference(image, debug_image )
			yield json.dumps(data) + "\n"
			##time.sleep(0.5)
	cam.release()
	cv.destroyAllWindows()
			      
@app.get("/video")
def video_feed():
    return StreamingResponse(video_inference(), media_type="text/event-stream")
	

@app.get("/datastream")
async def render_ds_page( ):
    async with aiofiles.open(r"static/datastream.html", mode="r") as f:
        return HTMLResponse(await f.read())


#@app.get("/")
#async def main():
#    async with aiofiles.open("app/static/index.html", mode="r") as f:
#        return HTMLResponse(await f.read())


@app.get("/")
async def main():
   async with aiofiles.open("app/static/home.html", mode="r") as f:
       return HTMLResponse(await f.read())
	
# Store image position and zoom state
image_state = {
    "imgX": 0,
    "imgY": 0,
    "zoom": 1
}


# API endpoint to get image state
@app.get("/image-state")
def get_image_state():
    return image_state

# API endpoint to update image state
@app.post("/update-image-state/")
def update_image_state(x: float, y: float, zoom: float):
    global image_state
    image_state["imgX"] = x
    image_state["imgY"] = y
    image_state["zoom"] = zoom
    return image_state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
