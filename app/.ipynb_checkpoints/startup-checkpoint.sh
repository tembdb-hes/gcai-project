#!/bin/bash


##start venv and navigate to app folder 
source mediapipe/bin/activate && cd fastapi/app

##populate 	ENVARS if running locally 

##ref for windows

##https://superuser.com/questions/1500272/equivalent-of-export-command-in-windows
##app directory
set APP_DIR=~/gest_rec_2024/fastapi/app

##device_params
set CAP_DEVICE=0
set CAP_WIDTH=640
set CAP_HEIGHT=480

##detection and tracking
set USE_STATIC_IMAGE_MODE=False
set MIN_DETECTION_CONFIDENCE=0.9  
set MIN_TRACKING_CONFIDENCE=0.5

##mode 
# 0 for no data capture, 1 for data_capture, see EXP_ACTION for labelling
set MODE=0


set MODEL_VERSION=2024_10_27_U

##model ref paths
set KP_CLASSIFIER_LABEL_PATH_0=modelv1/keypoint_classifier/keypoint_classifier_label.csv
set KP_CLASSIFIER_LABEL_PATH_2024_10_17=modelv1/keypoint_classifier_2024_10_17/keypoint_labels_z.csv
set KP_CLASSIFIER_LABEL_PATH_2024_10_21=modelv1/keypoint_classifier_20241021133457_32000/key_point_labels_20241021133457_32000.csv 
set KP_CLASSIFER_LABEL_PATH_2024_10_27_U=modelv1/keypoint_classifier_20241027010314_32000/keypoint_classifier_labels.csv
set PH_CLASSIFIER_LABEL_PATH=modelv1/point_history_classifier/point_history_classifier_label.csv 

##move/zoom Params
set XY_PX_CHG_PARAM=5
set ZOOM_PCT_CHG_PARAM=0.025
set LOCK_CHECK_LEN=20


##Data Save Paths
set POINT_HISTORY_DATA_FILE=modelv1/keypoint_classifier/point_history_z.csv 
set KEYPOINT_DATA_FILE=modelv1/keypoint_classifier/keypoint_z.csv       


##Inference Params
set NOISE_THRESH=0.45        
set STARTING_CURRENT_SCALE=1.0
set MIN_DISPLAY_SCALE=0.1  

##Verbose output
set VERBOSE=1      


## Data Collection Action (gesture when using vidoe to collect data , e.g open palm e.t.c)
set EXP_ACTION=None

## start the fast api app
uvicorn main:app --reload
 
