# in this file we can load static images from dataset and test accuracy of the program
# main goal is to test image enhancement and image processing functions

import cv2
import numpy as np
import os
import random
import time
import threading
import queue
import keyboard

import displayControl.display as displays
from imageProcessing.imageTools import process_base_image, preprocess_trainImg, get_features_sift, RESULTS_SIZE
from imageProcessing.imageTools import dmc_control, impurity_control, rentgen_control, pictograms_control
from tools import *


d = displays.DisplayProcessor()

results = [None] * RESULTS_SIZE


parameters = {
    # parameters for impurity control (used in base image processing)
    'base_dilatation_size': 25,
    # parameters for image enhancement
    'contrast': 2.0,
    'brightness': -10,
    'canny_low': 40,
    'canny_high': 110,
    'threshold': 60,
    'neighborhood_size': 91,
    'threshold_adjustment': 30,
    'surrounding': 35,
    # parameters for dmc control
    'shrink': 1,
    'deviation': 40,
    'threshold': 0,
    'timeout': 600,
    # parameters for pictograms control
    'pictograms_dilatation_size': 25,
    # parameters for rentgen control
    'pattern_threshold': 0.55,
    'canny_threshold_low': 40,
    'canny_threshold_high': 75,
    'join_iterations': 11,
    'surrounding_iterations': 6,
    'deform_pixels_by': 8
}




base_info = process_base_image(parameters=parameters)
dmc_code, inverted_dilated_base, base_data_dict_list, base_img, text_mask = base_info


# dataset path - there are two directories "bad" and "good" with images
print_dataset_path = "resources/print_box1_1"
rentgen_dataset_path = "resources/rentgen_box1_1"

# load images full paths from dataset to lists
print_images_good = os.listdir(print_dataset_path + "/good")
print_images_bad = os.listdir(print_dataset_path + "/bad")
rentgen_images_good = os.listdir(rentgen_dataset_path + "/good")
rentgen_images_bad = os.listdir(rentgen_dataset_path + "/bad")


# sort images by number in filename
print_images_good.sort(key=lambda x: int(x.split(".")[0]))
print_images_bad.sort(key=lambda x: int(x.split(".")[0]))
rentgen_images_good.sort(key=lambda x: int(x.split(".")[0]))
rentgen_images_bad.sort(key=lambda x: int(x.split(".")[0]))


# make full paths
print_images_good = [print_dataset_path + "/good/" + img for img in print_images_good]
print_images_bad = [print_dataset_path + "/bad/" + img for img in print_images_bad]
rentgen_images_good = [rentgen_dataset_path + "/good/" + img for img in rentgen_images_good]
rentgen_images_bad = [rentgen_dataset_path + "/bad/" + img for img in rentgen_images_bad]

len_good = len(print_images_good)
len_bad = len(print_images_bad)

output = {}

# test image enhancement
img = cv2.imread(print_images_good[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
enhanced_image = preprocess_trainImg(img)
# save enhanced image to file
cv2.imwrite("enhanced_image.jpg", enhanced_image)

# test image enhancement
def test_enhancement_bulk(paths_list):
    for img_path in paths_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_image = preprocess_trainImg(img)
        print("img_path: ", img_path)
        d.show_image(enhanced_image)
        d.active_wait()

# test_enhancement_bulk(print_images_good)

def test_dmc_control_bulk(paths_list, params={}, show_images=False):
    # možnost měnit parametry v decode funkci
        # shrink, deviation, threshold, timeout
        # shrink - zmenšení obrazu
        # timeout - časový limit pro dekódování - snížení doby zpracování, ale může vzdat detekci správného kódu
        # ostatní mohou ovlivnit rychlost
    # lokální výřez extrémně snižuje čas zpracování
    # zmenšení obrazu je vliv na rychlost a může pomoct při detekci kódu


    failed = 0
    corections = 0
    all = len(paths_list)

    result_labels = []
    # measure time
    start = time.time()

    for img_path in paths_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # look only on 900 - 1700 (rows - x axis) and 300 - 1000 (columns - y axis)
        img = img[300:1000, 900:1700]

        dmc_control(dmc_code, img, results, params)

        # if dmc control fails on image, show image
        if results[DMC][1] == (0,0,255):
            failed += 1
            print("DMC control failed on image: ", img_path)
            # d.show_image(img)
            # d.active_wait()
            # img_e = cv2.resize(img, (0,0), fx=0.8, fy=0.8)
            # dmc_control(dmc_code, img_e, results)
            # if results[DMC][1] == (0,0,255):
            #     print("DMC control failed on image after resizing: ", img_path)
            #     # d.show_image(img)
            #     # d.active_wait()
            # else:
            #     corections += 1
            #     print(COLOR_GREEN + "DMC control OK after resizing")
            #     print(COLOR_RESET)
            #     continue


            # try resize image
            # img_e = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            tmp_shrink = params['shrink']
            params['shrink'] = tmp_shrink+1
            dmc_control(dmc_code, img, results, params)
            params['shrink'] = tmp_shrink

            if results[DMC][1] == (0,0,255):
                print("DMC control failed on image after resizing: ", img_path)
                if show_images:
                    d.show_image(img)
                    d.active_wait()
                result_labels.append(False)
            else:
                corections += 1
                print(COLOR_GREEN + "DMC control OK after resizing")
                print(COLOR_RESET)
                result_labels.append(True)
                continue
        else:
            print(COLOR_GREEN + "DMC control OK")
            print(COLOR_RESET)
            result_labels.append(True)



    # time measurement
    end = time.time()
    print("Time elapsed: ", end - start , "s")
    print("time per image: ", (end - start) / all, "s")


    print("Failed: ", failed, " out of ", all)
    print("Corections: ", corections, " out of ", all)
    print("Success rate: ", (all - failed + corections) / all * 100, "%")
    return result_labels

output['dmc_good'] = test_dmc_control_bulk(print_images_good, params=parameters)
output['dmc_bad'] = test_dmc_control_bulk(print_images_bad, params=parameters)

def test_impurity_control_bulk(paths_list, show_images=False):
    result_labels = []
    for img_path in paths_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_image = preprocess_trainImg(img)

        src_pts, dst_pts = get_features_sift(base_img, enhanced_image)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            print("Homography matrix is None")
            d.show_image(enhanced_image)
            d.active_wait()
            continue
        impurity_control(inverted_dilated_base, enhanced_image, H, results)

        # if impurity control fails on good image, show image
        if np.mean(results[IMPURITY]) < 255:
            # erode results[IMPURITY]
            results[IMPURITY] = cv2.erode(results[IMPURITY], np.ones((3,3), np.uint8), iterations=1)
            # print("Impurity control failed on image: ", img_path)
            # make enhanced image with red impurities
            tmp = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
            
            # add impurity channel as red to tmp image
            tmp[:,:,0] = cv2.bitwise_and(tmp[:,:,0], results[IMPURITY])
            tmp[:,:,1] = cv2.bitwise_and(tmp[:,:,1], results[IMPURITY])
            tmp[:,:,2] = cv2.bitwise_or(tmp[:,:,2], cv2.bitwise_not(results[IMPURITY]))
            
            print(COLOR_RED)
            print("NOT OK")
            print(COLOR_RESET)

            if show_images:        
                d.show_image(tmp)
                # d.show_image(results[IMPURITY])
                d.active_wait()
            result_labels.append(False)
        else:
            # print ok in green then change back to default text color
            print(COLOR_GREEN)
            print("OK")
            print(COLOR_RESET)
            result_labels.append(True)
    return result_labels

output['impurity_good'] = test_impurity_control_bulk(print_images_good)
# output['impurity_bad'] = test_impurity_control_bulk(print_images_bad)

def test_pictograms_control_bulk(paths_list, params={}, show_images=False):

    result_labels = []
    for img_path in paths_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_image = preprocess_trainImg(img, messy_binarization=True)

        src_pts, dst_pts = get_features_sift(base_img, enhanced_image)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            print("Homography matrix is None")
            d.show_image(enhanced_image)
            d.active_wait()
            continue
        pictograms_control(base_img, enhanced_image, H, results, params)

        # if PICTOGRAMS control fails on good image, show image
        if np.mean(results[PICTOGRAMS]) < 255:
            # erode results[PICTOGRAMS]
            results[PICTOGRAMS] = cv2.erode(results[PICTOGRAMS], np.ones((3,3), np.uint8), iterations=1)
            # print("PICTOGRAMS control failed on image: ", img_path)
            # make enhanced image with red impurities
            tmp = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

            # add PICTOGRAMS channel as red to tmp image
            tmp[:,:,0] = cv2.bitwise_and(tmp[:,:,0], results[PICTOGRAMS])
            tmp[:,:,1] = cv2.bitwise_and(tmp[:,:,1], results[PICTOGRAMS])
            tmp[:,:,2] = cv2.bitwise_or(tmp[:,:,2], cv2.bitwise_not(results[PICTOGRAMS]))

            print(COLOR_RED)
            print("NOT OK")
            print(COLOR_RESET)

            if show_images:
                d.show_image(tmp)
                # d.show_image(results[PICTOGRAMS])
                d.active_wait()
            result_labels.append(False)
        else:
            # print ok in green then change back to default text color
            print(COLOR_GREEN)
            print("OK")
            print(COLOR_RESET)
            result_labels.append(True)
    return result_labels

output['pictograms_good'] = test_pictograms_control_bulk(print_images_good)
# output['pictograms_bad'] = test_pictograms_control_bulk(print_images_bad)

def test_rentgen_control_bulk(paths_list, params={}, show_images=False):
    result_labels = []
    for img_path in paths_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret = rentgen_control(img, results, params)
        if show_images:
            d.show_image(ret)
            d.active_wait()

        # if rentgen control fails on good image, show image
        if np.mean(results[RENTGEN]) < 255:
            # erode results[RENTGEN]
            results[RENTGEN] = cv2.erode(results[RENTGEN], np.ones((3,3), np.uint8), iterations=1)
            # print("Rentgen control failed on image: ", img_path)
            # make enhanced image with red impurities
            tmp = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)

            # add RENTGEN channel as red to tmp image
            tmp[:,:,0] = cv2.bitwise_and(tmp[:,:,0], results[RENTGEN])
            tmp[:,:,1] = cv2.bitwise_and(tmp[:,:,1], results[RENTGEN])
            tmp[:,:,2] = cv2.bitwise_or(tmp[:,:,2], cv2.bitwise_not(results[RENTGEN]))

            print(COLOR_RED)
            print("NOT OK")
            print(COLOR_RESET)
            if show_images:
                d.show_image(tmp)
                # d.show_image(results[RENTGEN])
                d.active_wait()
            result_labels.append(False)
        else:
            # print ok in green then change back to default text color
            print(COLOR_GREEN)
            print("OK")
            print(COLOR_RESET)
            result_labels.append(True)
    return result_labels

# output['rentgen_good'] = test_rentgen_control_bulk(rentgen_images_good)
# output['rentgen_bad'] = test_rentgen_control_bulk(rentgen_images_bad)


for i, img_path in enumerate(print_images_good):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # put text on img to show results
    img = cv2.putText(img, "DMC: " + str(output['dmc_good'][i]), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
    img = cv2.putText(img, "Impurity: " + str(output['impurity_good'][i]), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
    img = cv2.putText(img, "Pictograms: " + str(output['pictograms_good'][i]), (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
    
    d.show_image(img)
    d.active_wait()





# co udělat... TODO TODO
# bulk testy aby vracely list s výsledky (asi jen true false)
# pushnout na git a zkontrolovat že na gitu jsou data
# vypočítat úšpěšnost
    # počet správně určených
    # počet špatně určených pozitivních (false positive) (neprošlo to i když mělo)
    # počet špatně určených negativních (false negative) (prošlo to i když nemělo)
# vytvořit funkci pro zobrazení výsledků
    # zobrazit obrázek s chybami případně označený zeleně že prošel
# pouštět s jinými parametry







