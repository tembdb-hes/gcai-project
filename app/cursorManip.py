import pyautogui, sys 
##pyautogui.readthedocs.io
pyautogui.FAILSAFE = False
import os 
import numpy as np

cap_width  = os.environ.get("CAP_WIDTH")
cap_height = os.environ.get("CAP_HEIGHT")
cap_width  = int(cap_width)
cap_height = int(cap_height)

screenWidth, screenHeight = pyautogui.size()
print("Screen Resolution : ")
print(screenWidth, screenHeight)
x_move = screenWidth/cap_width
y_move = screenHeight/cap_height

###screen grid the center in box three by three , map to center ###
screen_grid = np.ones([screenHeight,screenWidth])


def move_cursor( x, y):
    currentMouseX, currentMouseY = pyautogui.position()
    print(currentMouseX, currentMouseY)
    print('====positions=======')
    print (int(x * 300 * x_move) ,int(y * 300 * y_move))
    
    x_pos = (int(x * 300 * x_move)// 20) * 20 
    y_pos = (int(y * 300 * y_move)// 20) * 20 
    
    pyautogui.moveTo( int(x * 300 * x_move) ,int(y * 300 * y_move))
    
    currentMouseX, currentMouseY = pyautogui.position()
    print(currentMouseX, currentMouseY)
  
  
def mouse_doubleClick(x=None ,y=None):
	if x is not None and y is not None:
	    pyautogui.doubleClick(x=x, y=y)	    
	else:
		pyautogui.doubleClick()
		

def mouse_click(x=None ,y=None):
	if x is not None and y is not None:
	    pyautogui.click(x=x, y=y)	    
	else:
		pyautogui.click()
		
		
def press_enter():
	pyautogui.press('enter')



def open_file_browser():
    None
	

