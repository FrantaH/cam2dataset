import re
import threading
import unicodedata
import cv2
import numpy as np

from pytesseract import image_to_string, image_to_data
from pylibdmtx.pylibdmtx import decode

from tools import *



ORIGINAL = 0
IMPURITY = 1
BASE = 2
RENTGEN = 3
DMC = 4
PICTOGRAMS = 5
OCR = 6

RESULTS_SIZE = 7



def load_baseImg(file_name='C:\\Users\\Uzivatel\\Desktop\\python_slider\\croped_base.jpg'):
    base_image = cv2.imread(file_name)
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    return base_gray

@time_measure
def process_base_image(file_name='C:\\Users\\Uzivatel\\Desktop\\python_slider\\resources\\base\\EMOXICEL_FAM_Trium_correct.jpg'):
# Read the image
    base_image = cv2.imread(file_name)
    text_image = cv2.imread(file_name.replace(".jpg","_text.jpg"))
    

    # Convert image to grayscale
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    text_gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    
    _, binary_image = cv2.threshold(base_gray, 127, 255, cv2.THRESH_BINARY)
    _, text_binary = cv2.threshold(text_gray, 127, 255, cv2.THRESH_BINARY)

    # Invert the colors
    inverted_base = cv2.bitwise_not(binary_image)
    
    # Decode DMC code
    decoded_objects = decode(binary_image, max_count=1)
    if len(decoded_objects) != 1:
        raise Exception("error, found different count of DMC codes:", len(decoded_objects))
    dmc_code = decoded_objects[0].data.decode('utf-8')

    # Dilate the inverted image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    inverted_dilated_base = cv2.dilate(inverted_base, kernel, iterations=3)
    
    # create text mask
    text_mask = cv2.erode(text_binary, kernel, iterations=4)
    # text_mask = cv2.bitwise_not(text_mask)

    # Extract text from the masked image
    # Get verbose data including boxes, confidences, line and page numbers
    image_data = image_to_data(text_binary, lang='eng+ita+fra+deu') # +ces
    

    # Split the string into lines
    lines = image_data.strip().split('\n')
    # Extract headers
    headers = lines[0].split('\t')
    # Create a list of dictionaries
    base_data_dict_list = []
    for line in lines[1:]:
        values = line.split('\t')
        row_dict = {header: value for header, value in zip(headers, values)}
        base_data_dict_list.append(row_dict)
    base_data_dict_list = cleanse_dict_base(base_data_dict_list)


    return dmc_code, inverted_dilated_base, base_data_dict_list, binary_image, text_mask

@time_measure
def preprocess_trainImg(img_image, display=False):
    # display=True
    # load image
    # img_gray = cv2.cvtColor(img_image, cv2.COLOR_BGR2GRAY)
    img_gray = img_image

    # d.show_image(img_gray)
    # Edge highlighting
    edges = cv2.Canny(img_gray, 100, 200)

    # Increase contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = -10    # Brightness control (0-100)
    enhanced_image = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)

    # d.show_image(enhanced_image)
    subtracted_image = cv2.subtract(enhanced_image, edges)
    # d.show_image(subtracted_image)



    gauss_filtered_image = cv2.GaussianBlur(subtracted_image, (3, 3), 0)
    # gauss_filtered_image = cv2.bilateralFilter(subtracted_image, 7, 11, 11)
    # gauss_filtered_image = cv2.medianBlur(subtracted_image, 7)
    # d.show_image(gauss_filtered_image,name="gauss_filtered_image")

    # d.show_image(gauss_filtered_image, name="gaussed")
    _, binarized_image = cv2.threshold(gauss_filtered_image, 170, 255, cv2.THRESH_BINARY)
    # binarized_image = cv2.adaptiveThreshold(gauss_filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 1)

    # d.show_image(binarized_image,name="binarized (result)")
    # Change brightness
    # brightness = 50
    # enhanced_image = np.where((255 - enhanced_image) < brightness, 255, enhanced_image + brightness)

    
    return binarized_image

@time_measure
def knn_brute_matching(des1, des2):

    # Create a BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:# and m.distance > 250:
            good_matches.append(m)
            # print(m.distance)

    return good_matches

def get_features_sift(base_img, train_img):
    # Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(base_img, None)
    kp2, des2 = sift.detectAndCompute(train_img, None)

    # Match descriptors
    # matches = knn_matching(des1, des2)
    matches = knn_brute_matching(des1, des2)


    # Extract corresponding keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # f_src_pts, f_dst_pts, filtered_indices = filter_outliners(src_pts, dst_pts)

    # new_kp1 = kp1[filtered_indices]
    # new_kp2 = kp2[filtered_indices]
    # new_good_matches = good_matches[filtered_indices]
    # matched_img = cv2.drawMatches(base_img, new_kp1, train_img, new_kp2, new_good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # matched_img = cv2.drawMatches(base_img, kp1, train_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # d.show_image(matched_img)

    return src_pts, dst_pts

def cleanse_word(word):
    word = unicodedata.normalize('NFC', word)  # Normalize to NFC Unicode
    no_diacritics = ''.join(c for c in unicodedata.normalize('NFD', word) if unicodedata.category(c) != 'Mn')
    lowercase_word = no_diacritics.lower()
    return lowercase_word

def cleanse_text_train(text):
    # Split text by new lines
    words = text.split()

    # Remove blank characters from each line and normalize to NFC Unicode
    cleaned_words = []
    for word in words:
        cleaned_words.append(cleanse_word(word))

    return cleaned_words

def cleanse_dict_base(text_dict_list):
    # Split text by new lines
    # words = text_dict.split()
    clean_dict_list = []
    for dict_item in text_dict_list:
        clean_text = cleanse_word(dict_item["text"])
        new_dict = {'left': int(dict_item["left"]), 'top': int(dict_item["top"]), 'width': int(dict_item["width"]), 'height': int(dict_item["height"]), 'text': dict_item["text"], 'clean_text': clean_text}
        if float(dict_item["conf"]) > 30.0:
            clean_dict_list.append(new_dict)


    clean_dict_list = sorted(clean_dict_list, key=lambda x: len(x["text"]), reverse=True)

    for item in clean_dict_list:
        no_special_chars = re.sub(r'[^a-z0-9-]', '.',  item["clean_text"]).strip('.')
        no_special_chars = re.sub(r'[\.]', '.?', no_special_chars)
        # add similar characters tolerance
        word_w_tolerance = re.sub(r'[il1t]', '[il1t]', no_special_chars)
        word_w_tolerance = re.sub(r'[ae8@gs56]', '[ae8@gs56]', word_w_tolerance)
        item["clean_text"] = re.sub(r'[obd]', '[obd]', word_w_tolerance)

    return clean_dict_list

@time_measure
def pictograms_control(base_img, train_img, H, results):
    #     potřebuju homography
    #     - kontrola piktogramů
    #         - zesílím vstup, invertuju a dám OR od transformed base
    #         - zkontroluju že je všude bílá

    # Dilate the inverted image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dilated_img = cv2.erode(train_img, kernel, iterations=3)
    inverted_img = cv2.bitwise_not(dilated_img)

    warped_base = cv2.warpPerspective(base_img, H, (train_img.shape[1], train_img.shape[0]), borderValue=255)

    or_img = cv2.bitwise_or(inverted_img, warped_base)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_or_img = cv2.erode(or_img, kernel, iterations=3)
    # d.show_image(warped_base, name="warped base")
    # d.show_image(inverted_img, name="image")
    # overlay = cv2.addWeighted(inverted_img, 0.5, warped_base, 0.5, 0)
    # d.show_image(overlay, name="overlay")
    # d.show_image(or_img, name="or_img")
    results[PICTOGRAMS] = dilated_or_img

    mean = np.mean(or_img)

    print("mean of pictograms_control = " + str(mean))

@time_measure
def dmc_control(dmc_code, train_img, results):
    decoded_objects = decode(train_img, max_count=1, timeout=2000)
    # color of result (red - wrong DMC)
    res_color = (0,0,255)
    shape_y, shape_x = train_img.shape
    rect_position = {'left': int(shape_x/2-100),'top': int(shape_y/2-100), 'right': int(shape_x/2+100), 'bottom': int(shape_y/2+100)}
    print(rect_position)
    results[DMC] = (rect_position, res_color)
    print(COLOR_RED)
    if len(decoded_objects) != 1:
        print("wrong count of DMC codes on image")
    else:
        if str(dmc_code) == str(decoded_objects[0].data.decode('utf-8')):
            res_color = (0,255,0)
            print(COLOR_GREEN)
            print("DMC is OK ...", res_color)
        else:
            print("DMC is NOT ok ...", res_color)

        print("input dmc:\t" + str(dmc_code))
        print("output dmc:\t" + str(decoded_objects[0].data.decode('utf-8')))
        # decoded_objects[0].rect=Rect(left=5, top=6, width=96, height=95))

        print(decoded_objects[0].rect.left)
        rect_position = {   'left': decoded_objects[0].rect.left,
                            'top': (shape_y - decoded_objects[0].rect.top - decoded_objects[0].rect.height), 
                            'right': (decoded_objects[0].rect.left + decoded_objects[0].rect.width), 
                            'bottom': (shape_y - decoded_objects[0].rect.top)} #  + decoded_objects[0].rect.height
        results[DMC] = (rect_position, res_color)
    print(COLOR_RESET)
    print(results[DMC])

@time_measure
def impurity_control(inverted_dilated_base, train_img, H, results):

    warped_base = cv2.warpPerspective(inverted_dilated_base, H, (train_img.shape[1], train_img.shape[0]))

    impurity_image = cv2.addWeighted(train_img, 1, warped_base, 1, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    impurity_image = cv2.erode(impurity_image, kernel, iterations = 3)
    


    # overlay = cv2.addWeighted(train_img, 0.5, impurity_image, 0.5, 0)
    # d.show_image(overlay, name="impurities (tecky) on packages")

    results[IMPURITY] = impurity_image
    print("impurity controll complete")

@time_measure
def OCR_control(base_dicts, train_img, text_mask, H, results):
    
    inverse_H = np.linalg.inv(H)
    normalised_train_img = cv2.warpPerspective(train_img, inverse_H, (text_mask.shape[1], text_mask.shape[0]))
    # d.show_image(normalised_train_img)
    # d.show_image(text_mask)
    normalised_train_img = cv2.bitwise_or(normalised_train_img,text_mask)
    # d.show_image(train_img)
    # Warp perspective using the inverse homography matrix
    

    time_measure_OCR = time_measure(image_to_string)
    # d.show_image(train_img)
    train_words = time_measure_OCR(normalised_train_img, lang='eng+fra+ita+deu') # +ces
    # train_text = image_to_string(normalised_train_img, lang='eng+fra+ita+ces+deu')
    train_words = cleanse_text_train(train_words)
    train_string = ''.join(train_words)
    isSame = True

    # if(len(base_lines) != len(train_lines)):
    #     isSame = False

    missed_words = []
    for item in base_dicts:
        train_string, N = re.subn(item["clean_text"], '', train_string, 1)
        if N != 1:
            # print("CHYBA: nenašel jsem jednu část z base")
            isSame = False
            missed_words.append(item)
        # print(word)
        # print(train_string)


    result_image = np.ones_like(text_mask) * 255
    
    import Levenshtein
    for miss in missed_words:

        distance = Levenshtein.distance(train_string, cleanse_word(miss["text"]), weights=(1,0,1))
        # print("miss: ", miss["text"])
        # print("distance: ", distance)
        cv2.rectangle(result_image, (miss["left"]-5, miss["top"]-5), (miss["left"]+miss["width"]+5,miss["top"]+miss["height"]+5), (0), 3)    
    print("text bin at the end was: ", train_string)

    results[OCR] = cv2.warpPerspective(result_image, H, (train_img.shape[1], train_img.shape[0]), borderValue=255)

    print("OCR controll complete, is same = " + str(isSame))

# zatím "pass"
@time_measure
def rentgen_control(results, img):
    # připojení k druhé kameře?
    results[RENTGEN] = cv2.resize(img,(960,540))
    print("rentgen control complete")

@time_measure
def get_homography(base_img, train_img):

    
    # get points matched to each other
    # src_pts, dst_pts = get_features_sift(base_img, train_img)
    src_pts, dst_pts = get_features_sift(base_img, train_img)

    # filter those which are way too off position
    # filtered_src_pts, filtered_dst_pts = filter_outliners(src_pts, dst_pts, distance_threshold = 50)


    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # M, _ = cv2.findHomography(src_pts, dst_pts, 0)



    # Warp base image onto img image
    warped_base = cv2.warpPerspective(base_img, H, (train_img.shape[1], train_img.shape[0]))

    # Overlay images
    # overlay = cv2.addWeighted(train_img, 0.5, warped_base, 0.5, 0)
    # overlay = cv2.subtract(train_img, warped_base)

    # Show overlay
    # d.show_image(overlay,"overlay")
    warped_base = cv2.cvtColor(warped_base.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return warped_base

@time_measure
def process_package(base_info, train_img, rentgen_img):
# def process_package(dmc_code, inverted_dilated_base, base_data_dict_list, base_img, train_img, text_mask, rentgen_img):

    dmc_code, inverted_dilated_base, base_data_dict_list, base_img, text_mask = base_info
    # split program into threads
    results = [None] * RESULTS_SIZE
    exceptions = []
    threads = []
    threads.append(threading.Thread(target=dmc_control, args=(dmc_code, train_img, results)))
    threads[-1].start()

    threads.append(threading.Thread(target=rentgen_control, args=(results,rentgen_img)))
    threads[-1].start()

    enhanced_img = preprocess_trainImg(train_img)

    # výpočet homografie
    get_features = time_measure(get_features_sift)
    src_pts, dst_pts = get_features(base_img, enhanced_img)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


    threads.append(threading.Thread(target=OCR_control, args=(base_data_dict_list, enhanced_img, text_mask, H, results)))
    threads[-1].start()
    threads.append(threading.Thread(target=pictograms_control, args=(base_img, enhanced_img, H, results)))
    threads[-1].start()
    threads.append(threading.Thread(target=impurity_control, args=(inverted_dilated_base, enhanced_img, H, results)))
    threads[-1].start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # merge OCR, PICTORGRAMS and IMPURITY errors
    result = cv2.min(results[PICTOGRAMS],results[OCR])
    result = cv2.min(result,results[IMPURITY])
    
    # make colored image combined with original image (red channel => errors)
    result = cv2.merge([
                        np.where(result<100, 0,  train_img),
                        np.where(result<100, 0,  train_img),
                        np.where(result<100, 255,train_img)])

    cv2.rectangle(result, (results[DMC][0]['left'],results[DMC][0]['top']), (results[DMC][0]['right'],results[DMC][0]['bottom']), results[DMC][1], 5)

    # combine OCR errors and PICTOGRAMS errors
    results[PICTOGRAMS] = cv2.min(results[PICTOGRAMS],results[OCR])

    # highlight errors and binarize them (OCR rects arent black (0) color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    results[PICTOGRAMS] = cv2.erode(results[PICTOGRAMS], kernel, iterations = 2)
    results[PICTOGRAMS] = cv2.threshold(results[PICTOGRAMS], 127, 255, cv2.THRESH_BINARY)[1]

    # resize down for canvas
    results[BASE] = cv2.resize(base_img, (960,540))
    results[ORIGINAL] = cv2.resize(train_img, (960,540))
    results[PICTOGRAMS] = cv2.resize(results[PICTOGRAMS], (960,540))
    results[IMPURITY] = cv2.resize(results[IMPURITY], (960,540))

    impurity_image = cv2.merge([
                        np.where(results[IMPURITY]==0,results[IMPURITY],results[ORIGINAL]),
                        np.where(results[IMPURITY]==0,results[IMPURITY],results[ORIGINAL]),
                        np.where(results[IMPURITY]==0,255,results[ORIGINAL])])

    print_image = cv2.merge([
                        np.where(results[PICTOGRAMS]==0,results[PICTOGRAMS],results[ORIGINAL]),
                        np.where(results[PICTOGRAMS]==0,results[PICTOGRAMS],results[ORIGINAL]),
                        np.where(results[PICTOGRAMS]==0,255,results[ORIGINAL])])


    # Make all images colored
    results[BASE] = cv2.cvtColor(results[BASE], cv2.COLOR_GRAY2BGR)
    results[ORIGINAL] = cv2.cvtColor(results[ORIGINAL], cv2.COLOR_GRAY2BGR)
    results[RENTGEN] = cv2.cvtColor(results[RENTGEN], cv2.COLOR_GRAY2BGR)

    # stack into 2x2 canvas
    left = np.vstack((print_image,results[BASE]))
    right = np.vstack((impurity_image,results[RENTGEN]))
    # left = np.vstack((results[ORIGINAL],results[BASE]))
    # right = np.vstack((results[IMPURITY],results[RENTGEN]))

    canvas = np.hstack((left,right))

    return result

