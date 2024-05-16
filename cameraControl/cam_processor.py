# -- coding: utf-8 --
# this file is based on "C:\Program Files (x86)\MVS\Development\Samples\Python\GrabImage\GrabImage.py"

import sys
# import threading
# import msvcrt
import numpy as np
# from ctypes import *
import cv2
import copy
import ctypes
# import time
from cameraControl.MvCameraControl_class import *
from pylibdmtx.pylibdmtx import decode
# import pytesseract
import math
import os
import time



def show_image(image, name="Image"):
    
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    # cv2.setMouseCallback(name, mouse_callback, {'window_name': name, 'click_fun': click_fun})
    
    # Move window to specified display (in this case to the left and down)
    cv2.moveWindow(name, -3000, 500)

    # Set window properties for full screen
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # cv2.resizeWindow(name, 1600, 900)  # Set your desired size here
    
    cv2.imshow(name, image)

    cv2.waitKey(1)
    # cv2.destroyAllWindows()

class CamProcessor:
    
    def __init__(self, serial, name, exposure_time = 0):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_USB_DEVICE
        self.name = name

        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print ("enum devices failed! ret[0x%x]" % ret)
            sys.exit()
        
        
        if deviceList.nDeviceNum == 0:
            print ("found no device!")
            sys.exit()

        # ch:创建相机实例 | en:Creat Camera Object
        self.cam = MvCamera()
        
        # get id by serial
        id = 0
        # for info in device list
        for i in range(deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            
            ptr = ctypes.cast(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber, ctypes.c_char_p)
            # Dereference the pointer and print the string
            serial_code = ptr.value.decode('ascii')
            if serial_code == serial:
                id = i
                print("camera %i with serial %s were chosen" % (i, serial_code))


        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[id], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print ("create handle fail! ret[0x%x]" % ret)
            sys.exit()


        # ch:打开设备 | en:Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print ("open device fail! ret[0x%x]" % ret)
            sys.exit()
        
        
        stBool = c_bool(False)
        ret = self.cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
        if ret != 0:
            print ("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print ("set trigger mode failed! ret[0x%x]" % ret)
            sys.exit()

        ret = self.cam.MV_CC_SetEnumValue("ExposureMode", MV_EXPOSURE_MODE_TIMED)
        if ret != 0:
            print ("set ExposureMode failed! ret[0x%x]" % ret)
            sys.exit()

        
        # set exposure time
        if (exposure_time!=0):
            # self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
        else:
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", 20000)
        if ret != 0:
            print ("set ExposureTime failed! ret[0x%x]" % ret)
            sys.exit()

        self.transform_matrix = []
        self.transform_matrix_inv = []

    def __del__(self):
        # Close device
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            print ("close deivce fail! ret[0x%x]" % ret)
            sys.exit()

        # Destroy handle
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print ("destroy handle fail! ret[0x%x]" % ret)
            sys.exit()

    def get_image(self):

        # en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print ("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        stOutFrame = MV_FRAME_OUT()  
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    
        ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            print(self.name + " get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
            
            # Convert the image data pointer to a numpy array
            img_data = np.ctypeslib.as_array(stOutFrame.pBufAddr, shape=(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
            img_data_copy = copy.deepcopy(img_data)
            # Display the image using OpenCV
            # cv2.imshow('image', img_data_copy)
            # cv2.waitKey(0)
            del img_data
            
            nRet = self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            
        else:
            print(self.name + " camera error - no data[0x%x]" % ret)
            sys.exit()

        # en:Stop grab image
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print ("stop grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        return img_data_copy
    
    def get_normalized_image(self):
        return normalize_img(self.get_image())

    def lookup_package(self):
        # Start grabbing images
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        # Variables to track motion
        prev_frame = None
        motion_counter = 0
        threshold = 2
        stop_motion_threshold = 4

        stOutFrame = MV_FRAME_OUT()  
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        
        while True:
            
            # Get the image buffer
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if stOutFrame.pBufAddr and ret == 0:
                # Convert the image data pointer to a numpy array
                img_data = np.ctypeslib.as_array(stOutFrame.pBufAddr, shape=(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
                # img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                # gauss_img_data = cv2.GaussianBlur(img_data, (3, 3), 0)
                gauss_img_data = normalize_img(img_data)
                # gauss_img_data = cv2.GaussianBlur(img_data, (5, 5), 0)
                if prev_frame is not None:
                    

                    # Compare current frame with the previous frame to detect motion
                    if mse(gauss_img_data, prev_frame) > threshold: # is in motion
                        motion_counter = 0
                        print(self.name + " is in motion - the diff is: ", mse(gauss_img_data, prev_frame))
                    elif mse(gauss_img_data, self.background) < threshold+5:  # is background
                        motion_counter = 0
                        print(self.name + " is background - the diff is: ", mse(gauss_img_data, prev_frame))
                        # sleep and let other threads more computation time
                        time.sleep(0.2)
                    else:                   # is not moving and is not background
                        motion_counter += 1
                        gauss_img_data = np.uint8((prev_frame * 0.5) + (gauss_img_data * 0.5))
                        print(self.name + " is on place - the diff is: ", mse(gauss_img_data, prev_frame))
                        
                    # If no motion is detected for some consecutive frames, assume the package has stopped moving
                    if motion_counter >= stop_motion_threshold:
                        print(self.name + " Package stopped moving.")
                        break

                prev_frame = gauss_img_data
                del img_data
                nRet = self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print("camera error - no data[0x%x]" % ret)
                sys.exit()
            time.sleep(0.4)


        # Stop grabbing images
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        return prev_frame  # Return the image of the stopped package
    
    def wait_for_background(self):
        
        while(True):
            time.sleep(0.2)
            new_image = self.get_normalized_image()
            dif = mse(new_image, self.background)
            # show_image(new_image)
            print(self.name + " diff is: ", dif)
            if dif < 2:  # is background
                # double check?
                # if mse(cv2.GaussianBlur(self.get_image(), (3, 3), 0), self.background) < 5:  # is background
                #     break
                break
            time.sleep(1)

    def capture_background(self):
        self.background = self.get_normalized_image()
        # self.background = cv2.GaussianBlur(self.background, (3, 3), 0)
        self.height, self.width = self.background.shape[:2]
        print(self.name + " shape of background (and camera resoltion): height:", self.height, " width:", self.width)
        return self.background
    
    def stream_vision(self):
        while True:
            sec = input("how many sec do you want to see video?:")
            if sec.isdigit():
                self.stream_vision_timed(sec)
            else:
                break

    def stream_vision_timed(self, sec):
        time_per_img = 500
        # en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print ("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        stOutFrame = MV_FRAME_OUT()  
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        for _ in range(int(float(sec)*1000/time_per_img)):
                
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if None != stOutFrame.pBufAddr and 0 == ret:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
                
                # Convert the image data pointer to a numpy array
                img_data = np.ctypeslib.as_array(stOutFrame.pBufAddr, shape=(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
                img_data_copy = copy.deepcopy(img_data)
                # Display the image using OpenCV
                # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('image', 1600, 900)
                cv2.imshow('Stream', img_data_copy)
                cv2.waitKey(time_per_img)
                # cv2.destroyAllWindows()
                del img_data
                
                nRet = self.cam.MV_CC_FreeImageBuffer(stOutFrame)
                
            else:
                print("camera error - no data[0x%x]" % ret)
                sys.exit()

        # cv2.destroyAllWindows()

        # en:Stop grab image
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print ("stop grabbing fail! ret[0x%x]" % ret)
            sys.exit()

    def get_pack_position(self, img):

        # this constant changes the rotation of the result (0 = oriented same as camera)
        ANGLE_OFFSET = 0


        # print("image")
        # show_image(img)
        # print("background")
        # show_image(self.background)

        # odečíst background od img
        # absolutní hodnota img
        # area_of_interest = cv2.absdiff(img, self.background)

        # print("area of interest (background subtract)")
        # show_image(area_of_interest)
        # cv2.waitKey(2000)
        # threshold img
        _, binarized_image = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)

        # show_image(binarized_image)
        # cv2.waitKey(2000)
        
        # kontury analýza
        contours, _ = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # filter small contours areas
        papers = [cnt for cnt in contours if cv2.contourArea(cnt) > 100000]
        dmc_mid = np.array([])
        pack_mid = np.array([])

        for paper in papers:
            # min bounding box (getting angle and position)
            minrect = cv2.minAreaRect(paper)
            box = cv2.boxPoints(minrect)
            box = np.intp(box)
            cv2.drawContours(binarized_image, [box], 0, (120), 4)  # Draw it on the image
            print("drawn contours and dmc")
            print(minrect)
            print(box)
            pack_mid = np.array([int(np.mean(box[:, 0])),int(np.mean(box[:, 1]))])


        # DMC analysis for geting rotation of package (without "text" analysis it is not possible to say upside down orientation)
        decoded_objects = decode(img, max_count=1, timeout=2000)

        print(len(decoded_objects))
        for obj in decoded_objects:
            print(f"Data: {obj.data.decode('utf-8')}, Position: {obj.rect}")
            top = img.shape[0] - obj.rect.top
            dmc_mid = np.array([int(obj.rect.left + obj.rect.width/2) , int(top - obj.rect.height/2)])


            # Draw rectangles around detected DMCs
            # cv2.rectangle(binarized_image, (int(obj.rect.left), top), 
            #             (int(obj.rect.left + obj.rect.width), int(top - obj.rect.height)),
            #             (120), 2)

            # Draw middle of DMC
            cv2.circle(binarized_image,dmc_mid,10,(120),4)

            # show_image(binarized_image)
        # return (binarized_image, pack_mid, 0)

        # draw middle of rectangle
        cv2.circle(binarized_image,pack_mid,10,(120),10)



        if len(dmc_mid) + len(pack_mid) < 3:
            raise ValueError("Unable to resolve either contour position or DMC position. dmc: " + str(dmc_mid) + "pack: " + str(pack_mid))


        # calculate angle related to DMC
        vec = dmc_mid - pack_mid
        approx_angle = math.atan2(vec[1],vec[0])
        # +145 to get 0° when package is well alligend
        # +180 to shift 0 rotation in the middle (rotation 360° and 0° -> 180) 
        # %360 to get interval 0-360 (possible subtraction of 180 to get 0 rotation on 0°, but this happens in robots.transform_angle_for_arm)
        approx_angle = (math.degrees(approx_angle) + 145 + ANGLE_OFFSET) % 360
        # print("pack degrees approximate: \t", approx_angle)
        pack_angle = minrect[2]


        # add correct rotation from approx_angle to more accurate measure pack_angle
        angle_diff = (approx_angle%90) - pack_angle
        if angle_diff > 45:
            pack_angle = (pack_angle + 90 + int(approx_angle/90)*90) % 360
        elif angle_diff < -45:
            pack_angle = (pack_angle - 90 + int(approx_angle/90)*90) % 360
        else:
            pack_angle = (pack_angle + int(approx_angle/90)*90) % 360

        print("dmc angle:  ", approx_angle)
        print("pack angle: ", pack_angle)

        # less accurate, faster calculation
        # pack_angle = approx_angle

        # show_image(binarized_image)

        return (binarized_image, pack_mid, pack_angle)

    def get_calibration_points(self):
        img = self.get_image()
        # optionally enhance or reduce size of image
        img = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)[1]
        # img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        
        # find keypoints on chessboard for calibration
        ret, corners = cv2.findChessboardCorners(img, (3,3))

        if ret == False :
            raise ValueError("Error: didnt find chessboard corners, calibration failed")
        

        # draw and label found points on image
        b = cv2.drawChessboardCorners(np.zeros_like(img), (3,3), corners, ret)
        for i, corner in enumerate(corners):
            cv2.putText(b,str(i),(int(corner[0][0]+10),int(corner[0][1]+10)),0,2,255,3)

        # draw points and labels as inverted color
        img = cv2.merge([
                        np.where(b>100, 255-img,  img),
                        np.where(b>100, 255-img,  img),
                        np.where(b>100, 255-img,  img)])

        # draw points and labels in green color
        # img = cv2.merge([
        #                 np.where(b>100, 0,  img),
        #                 np.where(b>100, 255,  img),
        #                 np.where(b>100, 0,  img)])
        
        # add image Z coordinate (0) and transform shit list [[[1,2,3]],[[1,2,3]]] to normal list [[1,2,3],[1,2,3]]
        corners = np.array([np.append(point[0],0) for point in corners])
        print(corners)

        return img, corners

    def compute_calibration(self, camera_points, robot_points):


        ret, self.transform_matrix, _ = cv2.estimateAffine3D(np.array(camera_points), np.array(robot_points))

        ret, self.transform_matrix_inv, _ = cv2.estimateAffine3D(np.array(robot_points), np.array(camera_points))
        print("estimate RET: ", ret)
        print("TRANSFORM MATRIX_inv: \n", self.transform_matrix_inv)
        print("TRANSFORM MATRIX: \n", self.transform_matrix)

        self.save_calib_matrix()

    def imagePoint_to_robotPoint(self, point):
        if len(point) == 2:
            point = np.append(point,0)#.reshape(-1,1)

        point = np.append(point,1).reshape(-1,1)
        # point = np.array(point).reshape(-1,1)

        if (len(self.transform_matrix) == len([])):
            return np.array([])
        # matmul
        print("matmul result: ", np.ravel(self.transform_matrix @ point))
        return np.ravel(self.transform_matrix @ point)[:3]

    def save_calib_matrix(self, filename="calibration_matrix.txt"):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        np.savetxt(filepath, self.transform_matrix)

    def load_calib_matrix(self, filename="calibration_matrix.txt"):
        print("loading calibration")
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(filepath):
            return False
        else:
            self.transform_matrix = np.loadtxt(filepath)
            print("calibration loaded")
            return True

def work_thread(cam=0, pData=0, nDataSize=0):
    stOutFrame = MV_FRAME_OUT()  
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
            
            # Convert the image data pointer to a numpy array
            img_data = np.ctypeslib.as_array(stOutFrame.pBufAddr, shape=(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
            # Display the image using OpenCV
            cv2.imshow('image', img_data)
            cv2.waitKey(0)
            del img_data
            # libc.free(data_pointer)
            
            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
            # g_bExit = True
        else:
            print("no data[0x%x]" % ret)
        # if g_bExit == True:
        #     break

def list_devices():
    
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_USB_DEVICE # MV_GIGE_DEVICE | 
    
    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print ("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print ("find no device!")
        sys.exit()

    print ("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print ("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print ("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)
            
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
#    print("MSE max: ", np.max(diff), "min:" , np.min(diff))
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

def normalize_img(img):
    # img = self.get_image()
    img = cv2.GaussianBlur(img, (3, 3), 0)
    mean = np.mean(img)


    img_normalized = (img - mean + 128)

    img_normalized = np.clip(img_normalized,0,255)

    img_normalized = np.uint8(img_normalized)

    return img_normalized
