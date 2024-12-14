##############################################
# Based on Mediapipe and Kazhuito00
# Created By Edwin Tembo, Jayden Chen
##############################################


import csv
import copy
import itertools
from   collections import Counter
from   collections import deque
import cv2 as cv
import numpy as np
import os
import numpy as np
import math
import warnings
import datetime
from datetime import datetime, timezone 
import base64
##from cursorManip import move_cursor,mouse_doubleClick, mouse_click

warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').disabled = True

from model_setup import InferenceModelParams

model_params = InferenceModelParams()

class ModelInference(InferenceModelParams):
	def __init__(self):
		super().__init__()
		
		
		
	def gen_scale_pos(self, pct_chg, pos_x_chg, pos_y_chg):
		"""
		  Generate the scaling factor
		"""
		scale_num = self.current_scale
		pos_x_num = self.current_pos_x
		pos_y_num = self.current_pos_y
		
		if pct_chg != 0. :
			scale_num  = self.current_scale + (pct_chg * (self.current_scale/2))
			
			
		pos_x_num = self.current_pos_x + pos_x_chg
		pos_y_num = self.current_pos_y + pos_y_chg
		
		if scale_num <0. :
			scale_num = self.min_display_scale
			
		scale = str(scale_num)
		pos_x = str(pos_x_num)
		pos_y = str(pos_y_num)
		
		self.current_scale = scale_num
		self.current_pos_x = pos_x_num
		self.current_pos_y = pos_y_num
		
		if self.debug:
		   print("========POSITIONS========")
		   print(scale, pos_x, pos_y, "\n")
		
		return scale, pos_x, pos_y
		
	def logging_csv(self, number, mode, landmark_list, landmark_list_z, point_history_list):

		if self.mode == 0:
			pass
		if self.mode == 1 :
			csv_path =  self.keypoint_file

			
			with open(csv_path, 'a', newline="") as f:
				writer = csv.writer(f)
				writer.writerow([number, *landmark_list, *landmark_list_z , self.exp_action])
		if self.mode == 2 and (0 <= number <= 9):
			csv_path = self.point_history_file
			
			with open(csv_path, 'a', newline="") as f:
				writer = csv.writer(f)
				writer.writerow([number, *point_history_list, self.exp_action])
				
		return
	
			
		
	def run_inference(self, image, debug_image):
                     
                           
                           fps = self.cvFpsCalc.get()
                        
                        
                           key = cv.waitKey(10)
                           if key == 9:  # TAB
                               return
                           number, mode = select_mode(key, self.mode)
                        
                           image.flags.writeable = False
                           results = self.hands.process(image)
                        
                           image.flags.writeable = True
                           

                       
                           if results.multi_hand_landmarks is not None :
                               for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                                       results.multi_handedness):
                                
                                   brect = calc_bounding_rect(debug_image, hand_landmarks)
                              
                                   landmark_list, landmark_list_z, unprocessed_landmark_list= calc_landmark_list(debug_image, hand_landmarks)
                                   
                             
                                   ## for pan history ##
                                   if self.pan_history.shape[0] == 0 :
                                       self.pan_history = np.array([unprocessed_landmark_list[8]])
                                   else:
                                       self.pan_history = np.concatenate((self.pan_history, [unprocessed_landmark_list[8]]) , axis = 0)[-5:, :]
                                       
                                   if self.debug:
                                       print('#################PAN HISTORY##################################')
                                       print(self.pan_history)
                                   
                                   pre_processed_landmark_list, pre_processed_landmark_list_z = pre_process_landmark(
                                       landmark_list, landmark_list_z, 1 if self.debug else 0)
                                       

                                   pre_processed_point_history_list = pre_process_point_history(
                                       debug_image, self.point_history)
                                       
                                   #logging
                                   self.logging_csv(number, mode, pre_processed_landmark_list,
                                                                  pre_processed_landmark_list_z,
                                                                  pre_processed_point_history_list, 
                                                   )
                        
                                   #hand_sign_id
                                   if self.model_version == '0':
                                       hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                                   else:
                                       hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list +  pre_processed_landmark_list_z)
                                       
                                   if self.debug:
                                      print("hand sign id " ,  hand_sign_id)    
                                      
                                   if hand_sign_id == 2 and self.model_version == '0':  

                                       self.point_history.append(landmark_list[8])  
                                   else:
                                       self.point_history.append([0, 0])
                        
                                   # Finger Point history 
                                   finger_gesture_id = 0
                                   point_history_len = len(pre_processed_point_history_list)
                                   if point_history_len == (self.history_length * 2):
                                       finger_gesture_id = self.point_history_classifier(
                                           pre_processed_point_history_list)
                        

                                   self.finger_gesture_history.append(finger_gesture_id)
                                   most_common_fg_id = Counter(
                                       self.finger_gesture_history).most_common()
                                   
                                 
                        
                                   if self.show_drawings:
                               
                                       debug_image = draw_bounding_rect(self.use_brect, debug_image, brect)
                                       debug_image = draw_landmarks(debug_image, landmark_list)
                                       debug_image = draw_info_text(
                                                debug_image,
                                                brect,
                                                handedness,
                                                self.keypoint_classifier_labels[hand_sign_id],
                                                self.point_history_classifier_labels[most_common_fg_id[0][0]],
                                                 )
                                    
                                   
                                   z_value = np.mean(landmark_list_z)
                                   
                                   pct_chg = 0.0
                                   if self.ct == 0:
                                       last_z = math.inf
                                       self.out_data_list = np.append(self.out_data_list,z_value) 

                                   else: 
                                      last_z = self.out_data_list[-1]
                                   
                                   if last_z == math.inf  or z_value is None or last_z is None :
                                       raw_pct_chg = 0.0
                                   else:
                                       raw_pct_chg =  0.0 if last_z == 0.0 else ((z_value - last_z)/last_z)
                                  
                                   try:
                                       kcl = self.keypoint_classifier_labels[hand_sign_id]
                                       
                                       if self.debug:
                                           print(f"Labels {self.keypoint_classifier_labels}")
                                           
                                   except Exception as e:
									   
                                       print(f"KCL HAND SIGN ERR : {e} , HAND_SIGN_ID {hand_sign_id}")
									   
                                   pch = self.point_history_classifier_labels[most_common_fg_id[0][0]]
                                   
                                  
                                       
                                   # to make this more user friendly 
                                   if self.last_kcl == kcl and abs(self.last_pct_chg) > 0:
                                       pct_chg = self.last_pct_chg
                                   else:
									   
                                       if   kcl in ["zoom_out_right", "zoom_out_left"]    : pct_chg = self.zoom_pct_chg_param
                                       elif kcl in [ "zoom_in_right", "zoom_in_left"] : pct_chg = -1 *  self.zoom_pct_chg_param 									   
                                       else: pct_chg = 0.
                                       
                                   if self.debug:    
                                       print(f"kcl : {kcl} - {raw_pct_chg} - {pct_chg}")
                                       
                                   pct_chg = 0.0 if str(pct_chg) == 'nan' else pct_chg 
                                   
                                   
                                   pos_x_chg = 0
                                   pos_y_chg = 0
                                   two_fingers_count = None
                                   
                                   
                                   if self.panning_mode in ["pointer" ,  "both"]:
									   
                                       if  'pointer_goleft' in kcl:
                                           pos_x_chg = -1 * self.xy_px_chg_param

                                       
                                       if  'pointer_goright' in kcl:
                                           pos_x_chg = self.xy_px_chg_param

                                       if  'pointer_up' in kcl:
                                           pos_y_chg = -1 * self.xy_px_chg_param

									   
                                       if  'pointer_down' in kcl:
                                           pos_y_chg = self.xy_px_chg_param
                                           
                                   if self.panning_mode in ["palm", "both"]:
									   
                                       
                                       if (self.pan_history.shape[0] > 1) and "palm" in kcl :
										   
                                            # Calculate difference between x coordinates and scale by self.pan_multiplier                                           
                                            pos_x_chg = int((self.pan_history[-1][0] - self.pan_history[-2][0])*self.pan_multiplier)
                                            
                                            # Calculate difference between y coordinates and scale by self.pan_multiplier 
                                            pos_y_chg = int((self.pan_history[-1][1] - self.pan_history[-2][1])*self.pan_multiplier)
                                            
                                   ## TOUCHLESS MODE 
                                   #if self.touchless_os_ops :
                                   #    if 'pointer' in kcl and self.inference_status  == 0:
										   
                                           ## move cursor
                                   #        if (self.pan_history.shape[0] > 1) :
                                   #            if self.debug:
                                   #                print("========printing pan history for ======")     
                                   #                                                    
                                   #            mean_pos = np.mean(self.pan_history[-3:], axis=0)

                                   #            move_cursor((mean_pos[0]), np.mean(mean_pos[1]))
                                           
                                   #    if 'fist' in kcl and self.inference_status  == 0:
                                                                           
                                   #        mouse_click()
                                                                             
                                       
                                   ####Locking######################
                                   if  'two_fingers' in kcl:

                                       two_fingers_count = np.sum(np.where((self.kcl_arr == 'two_fingers_left') | (self.kcl_arr =='two_fingers_right'), 1, 0))
                                       ###if the very last is this,  toggle lock on or off 
                                       
                                       if two_fingers_count == 0:
										   
                                           self.init_status_change = 1
                                           
                                           
                                       if self.init_status_change == 1 and two_fingers_count  >= self.lock_check_len:
										   										   
                                            self.inference_status   = abs(self.inference_status - 1)
                                            self.kcl_arr            = np.array([])
                                            self.init_status_change = 0
                                   else:
										   
                                       self.init_status_change  = 0
                                       self.kcl_arr= np.delete(self.kcl_arr ,np.where((self.kcl_arr == 'two_fingers_left') | (self.kcl_arr =='two_fingers_right')))
                                       
                                   ##reset the transformation params        									   
                                   scale = None
                                   pos_x = None
                                   pos_y = None

				   ## Change inference status 					       
                                   if self.inference_status == 1 :
                                       scale,pos_x, pos_y = self.gen_scale_pos(pct_chg, pos_x_chg, pos_y_chg)
                                   
                                   self.kcl_arr = np.append(self.kcl_arr, kcl)[ -1 * self.lock_check_len:]  
                                   if self.debug:
                                       print("===KCL arr====")
                                       print(f"Two Finger Count {two_fingers_count}")
                                       print(self.kcl_arr, "\n")
                                       print("inference status" , self.inference_status, "\n")
                                   
                                   ## draw info on image 
                                   debug_image = draw_info(debug_image, fps, mode, number)

			                       
                                   if self.inference_status == 0: 
				       #save the output data
                                       out_data = {"scale"  : self.current_scale , 
                                                   "pos_x"  : self.current_pos_x, 
                                                   "pos_y"  : self.current_pos_y, 
                                                   "fps"    : fps , 
                                                   "kcl"    : "n/a",
												   "status" :"Locked", 
												   "gesture": self.ui_labels.get(kcl).get("gesture"),
										           "action" : self.ui_labels.get(kcl).get("action")

											      }
                                   else:
				       #save the ouptut data    
                                       out_data = {"scale" : scale , 
                                                   "pos_x" : pos_x , 
                                                   "pos_y" : pos_y, 
                                                   "fps"   : fps , 
                                                   "kcl"   : kcl , 
                                                   "status": "Active", 
                                                   "gesture": self.ui_labels.get(kcl).get("gesture"),
										           "action" : self.ui_labels.get(kcl).get("action")
										   
										           
                                                   }
                                              
                                   if self.collect_exp_data:
                                       self.collect_landmark_data(self.ct, out_data)

                                   self.ct+=1
                                   ##this only works for fixed non-zero pct_chg
                                   self.last_kcl = kcl 
                                   self.last_pct_chg = pct_chg
                                   
                               self.inactivity_start = None;

                           else:
                               #inactivity lock
                               if self.inactivity_start is None:
                                   self.inactivity_start = datetime.now(tz=timezone.utc)
                               status = 'n/a'
                               if self.inactivity_start is not None and self.inactivity_limit_sec is not None:
                                   inactivity_duration = datetime.now(tz=timezone.utc) - self.inactivity_start
                                   self.inference_status = 0 if inactivity_duration.seconds  > self.inactivity_limit_sec else self.inference_status
                                   status = 'Locked' if inactivity_duration.seconds  > self.inactivity_limit_sec else 'n/a'
								   
							   
                               self.point_history.append([0, 0])
                               out_data = {"scale" : self.current_scale , 
                                           "pos_x" : self.current_pos_x, 
                                           "pos_y" : self.current_pos_y, 
                                           "fps"   : fps , 
                                           "kcl"   : "n/a",
										   "status": status, 
										   "gesture": "n/a",
										   "action" : "n/a"
										  }
                                              
                               self.kcl_arr=np.delete(self.kcl_arr , np.where((self.kcl_arr == 'two_fingers_left') | (self.kcl_arr =='two_fingers_right')))
                               self.last_kcl = 'n/a'
                               self.last_pct_chg = 0.
                               ##$if self.last_kcl != 'n/a':
                               self.kcl_arr = np.append(self.kcl_arr, 'n/a')[-1 * self.lock_check_len:]   
                               
                               if self.debug:
                                   print("===KCL arr====")
                                   print(self.kcl_arr, "\n")
                                   print("inference status" , self.inference_status, "\n")
                                   print("status: " , status)
                                          
                               if self.collect_exp_data:
                                   self.collect_landmark_data(self.ct, out_data)
                                   
                                   
                               if self.show_drawings:                               
                                   debug_image = draw_point_history(debug_image, self.point_history)
                                   debug_image = draw_info(debug_image, fps, mode, number)
                           debug_image = cv.cvtColor(debug_image, cv.COLOR_RGB2BGR)
                           _, buffer = cv.imencode('.jpg', debug_image)
                           frame = base64.b64encode(buffer).decode('utf-8')		               

                  

                           return frame,out_data
                        



def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    """
     Bounding Box
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    """
     Collect and prepare landmarks
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point   = []
    landmark_point_z = []
    temporary_point  = []

  
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        temporary_point.append([landmark.x, landmark.y])

        landmark_point.append([landmark_x, landmark_y])
        landmark_point_z.append(landmark_z)
      
    return landmark_point, landmark_point_z, temporary_point


def pre_process_landmark(landmark_list, landmark_list_z, verbose):
    temp_landmark_list = copy.deepcopy(landmark_list)
    temp_landmark_list_z = copy.deepcopy(landmark_list_z)

   
    base_x, base_y, base_z = 0, 0 , 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y= landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        
    for index_z, landmark_point_z in enumerate(temp_landmark_list_z):
        if index_z == 0:
            base_z = landmark_point_z

        temp_landmark_list_z[index_z] = temp_landmark_list_z[index_z] - base_z
       

    
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
        
    if  verbose == 1 :
        print("====orig_landmark_list_z=====")
        print(landmark_list_z)      
        print("====temp_landmark_list_z=====")
        print(temp_landmark_list_z)   


   
    max_value = max(list(map(abs, temp_landmark_list)))
    
    max_value_z = max(map(abs, temp_landmark_list_z))

    def normalize_(n):
        return n / max_value
        
    def normalize_z_(n):
        return n / max_value_z

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    temp_landmark_list_z = list(map(normalize_z_, temp_landmark_list_z))

    return temp_landmark_list, temp_landmark_list_z


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

  
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history




def draw_landmarks(image, landmark_point):
    
    if len(landmark_point) > 0:
        
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

       
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

       
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16 :
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text , show_finger_text = False):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "" and show_finger_text:
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    pass
