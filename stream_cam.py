
import cv2
import os
import cameraControl.cam_processor as cameras
import serial
import time
import displayControl.display as displays
import keyboard


# open and configure the serial port
ser = serial.Serial("COM4", 9600, timeout=1)
time.sleep(2)


def open_gate_ok():
    ser.write(b"AT+O1")

def open_gate_fail():
    ser.write(b"AT+O2")

def close_gate():
    ser.write(b"AT+AC")



#open_gate_ok()
#time.sleep(2)
#close_gate()
#time.sleep(2)
#open_gate_fail()
#time.sleep(2)
close_gate()


# open and connect to the camera
camera_print = cameras.CamProcessor("DA1274571", "print_checker",crop_background=500)
camera_rentgen = cameras.CamProcessor("DA3215511", "rentgen")

d = displays.DisplayProcessor()

def on_press(event):
    if(event.name == 'q'):
        print("press control + c to exit")
    elif(event.name == 'c'):
        close_gate()
        print("Closing the gate")
    elif(event.name == 'o'):
        open_gate_ok()
        print("Opening the gate")
    elif(event.name == 'f'):
        open_gate_fail()
        print("Opening the gate")

        



keyboard.hook(on_press)


input("Press enter to capture the background")
camera_print.capture_background()
d.show_image(camera_print.background)


# stream vision

c = input("what camera you want to see? r = rentgen, otherwise printchecker")
if(c=='r'):
    showing_cam = camera_rentgen
else:
    showing_cam = camera_print

while(True):
    d.show_image(showing_cam.get_rgb_image())
    time.sleep(0.5)
