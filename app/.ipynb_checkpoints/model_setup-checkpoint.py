##############################################
# Based on Mediapipe and Kazhuito00
# Created By Edwin Tembo
##############################################

import time 
import csv
import copy
import argparse
import itertools
from   collections import Counter
from   collections import deque
import numpy as np
import mediapipe as mp
import os

import pandas as pd  
import datetime

from utilsv1 import CvFpsCalc
from modelv1 import PointHistoryClassifier   
    
class DotDict(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


def envar_type_checks(item, var_name,  expected_type, expected_range=None, allowed_values=None):
	
     try:

        if expected_type == int:
            item = int(item)
        elif expected_type == float:
            item = float(item)
        elif expected_type == bool:
            item =False if item.lower() == 'false' or item is None  else True
				 
			
     except TypeError:
            raise TypeError(f"{var_name} must be an {expected_type}")
            
            
     if expected_range is not None:
         assert  min(expected_range) <= item <= max(expected_range), f"{var_name} must be an integer between {min(expected_range)} and {max(expected_range)}"
		
     if allowed_values is not None:
         assert item in allowed_values, f"{var_name} must be one of these values: {allowed_values}"
		
     return item
	


class InferenceModelParams():
    def __init__(self):
        self.cap_device               = os.environ["CAP_DEVICE"]
        
        self.cap_device               = envar_type_checks(item =self.cap_device, 
                                                          var_name ="CAP_DEVICE",  
                                                          expected_type = int, 
                                                          expected_range= None, 
                                                          allowed_values=[0,1]
                                                          )
        
        self.cap_width                = os.environ["CAP_WIDTH"]
        self.cap_width                = envar_type_checks(item =self.cap_width, 
                                                          var_name ="CAP_WIDTH",  
                                                          expected_type = int, 
                                                          expected_range= [240,2160],
                                                          allowed_values=None
                                                          )
        
        self.cap_height               = os.environ["CAP_HEIGHT"]
        self.cap_height              = envar_type_checks(item =self.cap_height, 
                                                          var_name ="CAP_HEIGHT",  
                                                          expected_type = int, 
                                                          expected_range= [320,4096] ,
                                                          allowed_values=None
                                                          )
        
        self.use_static_image_mode    = os.environ["USE_STATIC_IMAGE_MODE"]
        self.use_static_image_mode    = envar_type_checks(item =self.use_static_image_mode, 
                                                          var_name ="USE_STATIC_IMAGE_MODE",  
                                                          expected_type = bool, 
                                                          expected_range= None,
                                                          allowed_values=None
                                                          )
        
        self.min_detection_confidence = os.environ["MIN_DETECTION_CONFIDENCE"]
        self.min_detection_confidence = envar_type_checks(item =self.min_detection_confidence, 
                                                          var_name ="MIN_DETECTION_CONFIDENCE",  
                                                          expected_type = float ,
                                                          expected_range= [0.1, 0.99],
                                                          allowed_values=None
                                                          )
        
        
        self.min_tracking_confidence  = os.environ["MIN_TRACKING_CONFIDENCE"]
        self.min_tracking_confidence  = envar_type_checks(item =self.min_tracking_confidence , 
                                                          var_name ="MIN_TRACKING_CONFIDENCE",  
                                                          expected_type = float ,
                                                          expected_range= [0.1, 0.99],
                                                          allowed_values=None
                                                          )

        self.mode                     = 0 if os.environ["MODE"] is None else os.environ["MODE"]
        self.mode                     = envar_type_checks(item =self.mode , 
                                                          var_name ="MODE",  
                                                          expected_type = int,
                                                          expected_range=None,
                                                          allowed_values=[0,1]
                                                         )
        
        print (f"The mode is {self.mode}" )
        self.model_version            = os.environ["MODEL_VERSION"]
        self.use_brect                = True
        self.show_drawings            = False
        self.max_hands                = 1
        
        ##Keypoint Label Models#############
        if self.model_version in ['0', None] :
            self.kp_classifier_label_path = os.environ["KP_CLASSIFIER_LABEL_PATH_0"]

            from modelv1 import KeyPointClassifier

        elif self.model_version == '2024_10_17':
            self.kp_classifier_label_path  = os.environ["KP_CLASSIFIER_LABEL_PATH_2024_10_17"]

            from modelv1 import KeyPointClassifier_2024_10_17 as KeyPointClassifier
                
        elif self.model_version == '2024_10_21':
			
             self.kp_classifier_label_path  = os.environ["KP_CLASSIFIER_LABEL_PATH_2024_10_21"]
             from modelv1 import KeyPointClassifier_20241021133457_32000 as KeyPointClassifier 
             
        elif self.model_version == '2024_10_27_U':
			
             self.kp_classifier_label_path  = 	os.environ["KP_CLASSIFER_LABEL_PATH_2024_10_27_U"]
             from modelv1 import KeypointClassifier_20241027010314_32000 as KeyPointClassifier 	
             

        self.ph_classifier_label_path  = os.environ["PH_CLASSIFIER_LABEL_PATH"]
        
        ##Output options:
        self.verbose = os.environ.get("VERBOSE") 
        assert self.verbose.isdigit() and self.verbose in ('0','1') , "VERBOSE must be the integers 0 or 1"
        self.verbose                   = int(os.environ.get("VERBOSE") )
        
        ##Toggle locking on/off
        self.kcl_arr                   = np.array([])

        self.init_status_change        = 0 
        self.inference_status          = 0
        
        self.lock_check_len            = os.environ["LOCK_CHECK_LEN"]
        
        try:
            self.lock_check_len = int(self.lock_check_len)
			
        except TypeError:
            raise TypeError("LOCK_CHECK_LEN must be an integer >=20")
						
        assert  20 <= self.lock_check_len  , "LOCK must be an integer greater then 20"
        
        ##Move/zoom params
        self.xy_px_chg_param           = os.environ["XY_PX_CHG_PARAM"]
        
        try:
            self.xy_px_chg_param = int(self.xy_px_chg_param)
			
        except TypeError:
            raise TypeError("XY_PX_CHG_PARAM must be an integer between 1 and 50")
						
        assert  1 <= self.xy_px_chg_param <= 50 , "XY_PX_CHG_PARAM must be an integer between 1 and 50"
	    
	    
        self.zoom_pct_chg_param           = os.environ["ZOOM_PCT_CHG_PARAM"]
        
        try:
            self.zoom_pct_chg_param  = float(self.zoom_pct_chg_param)
			
        except TypeError:
            raise TypeError("ZOOM_PCT_CHG_PARAM must be an floating point number between 0.01 and 0.20")
				
        assert  0.01 <= self.zoom_pct_chg_param<= 0.20 , "ZOOM_PCT_CHG_PARAM must be an floating point number between 0.01 and 0.20"
        
        #Data Collection Paths
        self.point_history_file        = os.environ["POINT_HISTORY_DATA_FILE"]        
        self.keypoint_file             = os.environ["KEYPOINT_DATA_FILE"]
   
        self.out_data_list             = np.array([])
        
        #Inference Params
        self.last_kcl                  = 'n/a'
        self.last_pct_chg              = 0.0
        
        
        try:
            self.noise_thresh              = float(os.environ.get("NOISE_THRESH"))
        except TypeError:
            raise TypeError("The NOISE_THRESH is a floating point number e.g. 0.54")
			
        self.noise_thresh              = 0.45 if self.noise_thresh is None else self.noise_thresh
        assert 0 < self.noise_thresh < 1.0 , 'Noise threshold is a fraction between 0 and 1'
        
        
        try:
            self.current_scale             = float(os.environ.get("STARTING_CURRENT_SCALE"))
        except TypeError:
            raise TypeError("The STARTING_CURRENT_SCALE is a floating point number e.g. 1.0")	
            
        
        self.current_pos_x             = 0
        self.current_pos_y             = 0
		   		   
		    
        self.current_scale                 = 1.0 if self.current_scale is None else self.current_scale
        assert 0.1 <= self.current_scale <= 1.0 , 'Starting current_scale must be is a floating point number between 0.1 and 1.0'
        
         
        try:
            self.min_display_scale         = float(os.environ.get("MIN_DISPLAY_SCALE"))
        except TypeError:
            raise TypeError("The MIN_DISPLAY_SCALE is a floating point number > 0 and < 1.0 e.g. 0.1")
             
             
        self.min_display_scale         = 0.1 if self.min_display_scale is None else self.min_display_scale       
        assert 0.1 <= self.min_display_scale <= 1.0 , 'Starting min_display_scale must be is a floating point number between 0.1 and 1.0'
        
        
        #Experimental Data Collection
        self.collect_exp_data  = False #TODO: remove 
        
        if int(self.mode) in [1,2]:
            self.exp_action        =  None if os.environ.get("EXP_ACTION") in (None, 'None') else os.environ.get("EXP_ACTION")
           
            print("The data collection action is : ", self.exp_action)
			
        else:
            self.exp_action    = None
			
        self.ct                = 0 # frame count init 
        
        #Ground truth values        
        
        self.gt_gesture        = None    #closed, pointer
        self.gt_side           = None    #left, right, hand 
        self.gt_orient         = None    #"facing_in" "facing_out" facing_in means palm_side towards screen
        
        
        self.exp_collect_data = False #TODO: remove
        if self.exp_action is None  and self.exp_collect_data:
          raise ValueError("exp_action is required for experimental data collection")

        #DataFrame for data collection :TODO - Remove
        self.df        = None
        ts             = time.time_ns()
        self.file_path = f"data/exp_data/{self.exp_action}_{self.gt_side}_{self.gt_gesture}_{self.gt_orient}_data_started{ts}.csv"
      
        #Models 
        self.mp_hands      = mp.solutions.hands
        self.hands         = self.mp_hands.Hands(
                                  static_image_mode =self.use_static_image_mode,
                                  max_num_hands     = self.max_hands,
                                  min_detection_confidence=self.min_detection_confidence,
                                  min_tracking_confidence =self.min_tracking_confidence,
                                 )         
         
        print(f"+++++++ Loading Models : {datetime.datetime.now()} ++++++++++")
        
        self.keypoint_classifier      = KeyPointClassifier()

        self.point_history_classifier = PointHistoryClassifier()
        
        print(f"++++++ End Loading Models : {datetime.datetime.now()} ++++++++++")
        
        #Labels
        with open(self.kp_classifier_label_path ,
              encoding='utf-8-sig') as f:
              self.keypoint_classifier_labels = csv.reader(f)
              self.keypoint_classifier_labels = [
              row[0] for row in self.keypoint_classifier_labels
              ]
              
        print(f"LABELS : {self.keypoint_classifier_labels}")
        
        with open(
            self.ph_classifier_label_path ,
            encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
            row[0] for row in self.point_history_classifier_labels
             ]

        #FPS class init
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        
        #Point History 
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)


        
    #Data
    def collect_landmark_data( self, frame_count, data):
								
        new_ts    = { "time_ns": [time.time_ns()]}
        new_data  = new_ts | data
        
        if frame_count == 0 :
            columns = list(new_data.keys())
            self.df = pd.DataFrame(data= new_data, columns = columns)
        else:
			
            inner_df  = pd.DataFrame(data = new_data, columns = list(new_data.keys()))
            self.df   = pd.concat([self.df, inner_df] , axis = 0)
        
        #Checkpoint 
        if frame_count%200 == 0 :
            self.df = self.df.sort_values(by ="time_ns").reset_index(drop =True)
            self.df.to_csv(self.file_path, index=False)
			
   
        

    
