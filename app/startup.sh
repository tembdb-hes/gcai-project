#!/bin/bash


##Change these appropriately
export VENV_LOCATION={{your_venv_location}} ##e.g.~/Documents
export APP_FULL_PATH={{full_path_to_app_dir}}  ##e.g~/gcai/main/app
export VENV_NAME={{your_venv_name}}  ##e.g. gcai_prototype

##start venv and navigate to app folder 
cd $VENV_LOCATION && source $VENV_NAME/bin/activate

cd $APP_FULL_PATH

##populate 	ENVARS if running locally 

export DEBUG=False

##User session expiration
export ACCESS_TOKEN_EXPIRE_MINUTES=1440

## for touchless capabilities (not Fully Functional)
export TOUCHLESS_OS_OPS=False

## Number of seconds until automatic inactivity lock for gesture recognition
export INACTIVITY_LIMIT_SEC=10

##the relative or full path to application dir. This file must run in the same location
export APP_DIR=./

##device_params
export CAP_DEVICE=0
export CAP_DEVICE2=2
export CAP_WIDTH=320
export CAP_HEIGHT=240

##detection and tracking
export USE_STATIC_IMAGE_MODE=False
export MIN_DETECTION_CONFIDENCE=0.9  
export MIN_TRACKING_CONFIDENCE=0.5
export PANNING_MODE=both
export PANNING_MULTIPLIER=500
export SHOW_INFERENCE_DRAWINGS=True
##mode 
# 0 for no data capture, 1 for data_capture, see EXP_ACTION for labelling
export MODE=0


## use MODEL_VERSION=2024_10_27_U for other model
export MODEL_VERSION=2024_11_24_O
##model ref paths
export KP_CLASSIFIER_LABEL_PATH_0=modelv1/keypoint_classifier/keypoint_classifier_label.csv
export KP_CLASSIFIER_LABEL_PATH_2024_10_17=modelv1/keypoint_classifier_2024_10_17/keypoint_labels_z.csv
export KP_CLASSIFIER_LABEL_PATH_2024_10_21=modelv1/keypoint_classifier_20241021133457_32000/key_point_labels_20241021133457_32000.csv 
export KP_CLASSIFER_LABEL_PATH_2024_10_27_U=modelv1/keypoint_classifier_20241027010314_32000/keypoint_classifier_labels.csv
export KP_CLASSIFER_LABEL_PATH_2024_11_24_O=modelv1/keypoint_classifier_20241124191842_32000_oversampling/keypoint_classifier_labels.csv
export PH_CLASSIFIER_LABEL_PATH=modelv1/point_history_classifier/point_history_classifier_label.csv 

##move/zoom Params
export XY_PX_CHG_PARAM=5
export ZOOM_PCT_CHG_PARAM=0.025
export LOCK_CHECK_LEN=20


##Data Save Paths
export POINT_HISTORY_DATA_FILE=modelv1/keypoint_classifier/point_history_z.csv 
export KEYPOINT_DATA_FILE=modelv1/keypoint_classifier/keypoint_z.csv       


##Inference Params
export NOISE_THRESH=0.45        
export STARTING_CURRENT_SCALE=1.0
export MIN_DISPLAY_SCALE=0.1  

## use 0 for NO SSL or 1 FOR SSL 
export USE_SSL=1

## CERTIFICATE LOCATION (Only needed when USE_SSL =1) - Change this to appropriate secure location in PRODUCTION 
export SSL_KEYFILE_PATH=./localhost.key 
export SSL_CERTFILE_PATH=./localhost.crt
     


## Data Collection Action (gesture when using vidoe to collect data , e.g open palm e.t.c)
export EXP_ACTION=None

## Run the application with or without SSL. For production, remove the reload arg.
if [ $USE_SSL -ge 1 ]; then 

     uvicorn main:app  --reload --ssl-keyfile $SSL_KEYFILE_PATH  --ssl-certfile $SSL_CERTFILE_PATH
else

     uvicorn main:app  --reload 
     
fi
