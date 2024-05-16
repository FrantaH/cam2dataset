import cv2
import numpy as np
import time
from pytesseract import image_to_string
import pytesseract
from pylibdmtx.pylibdmtx import decode
import threading
import re
import unicodedata
import cameraControl.cam_processor as cameras
import robotControl.robot_processor as robots
import displayControl.display as displays
import queue

print("fuck")

DEBUG=True


ORIGINAL = 0
IMPURITY = 1
BASE = 2
RENTGEN = 3
DMC = 4
PICTOGRAMS = 5
OCR = 6

RESULTS_SIZE = 7


# ANSI escape codes for text colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

# variable for display - global for other functions
d = None

# sync var for threads
checker_free = None

# robot var
rob = None

image_queue = queue.Queue()

def time_measure(fce):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = fce(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Processing time for function:", fce.__name__, "was:", elapsed_time, "seconds")
        return result
    return wrapper

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
    # dmc_code = 'dmc cod'

    # Dilate the inverted image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    inverted_dilated_base = cv2.dilate(inverted_base, kernel, iterations=3)
    
    # create text mask
    text_mask = cv2.erode(text_binary, kernel, iterations=4)
    # text_mask = cv2.bitwise_not(text_mask)
    # d.show_image(text_mask)

    # Extract text from the masked image
    # Get verbose data including boxes, confidences, line and page numbers
    image_data = pytesseract.image_to_data(text_binary, lang='eng+ita+fra+deu') # +ces
    

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

    # Display enhancements
    if display:
        d.show_images([img_gray,edges,binarized_image])
    
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
            print("CHYBA: nenašel jsem jednu část z base")
            isSame = False
            missed_words.append(item)
        # print(word)
        # print(train_string)


    result_image = np.ones_like(text_mask) * 255
    
    import Levenshtein
    for miss in missed_words:

        distance = Levenshtein.distance(train_string, cleanse_word(miss["text"]), weights=(1,0,1))
        print("miss: ", miss["text"])
        print("distance: ", distance)
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
def process_package(dmc_code, inverted_dilated_base, base_data_dict_list, base_img, train_img, text_mask, rentgen_img):


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




if __name__ == '__main__':

    print("      I N I T    R O B O T      ")
    print("--------------------------------")

    rob = robots.RobotProcessor()

    print("    I N I T    C A M E R A S    ")
    print("--------------------------------")


    d = displays.DisplayProcessor()

    # list cameras info
    # cameras.list_devices()

    # connect to cam by serial number of camera and create that camera object
    # DA1274571
    # 00K81747629
    print_check_camera = cameras.CamProcessor("DA1274571", "print_checker")

    sorter_camera = cameras.CamProcessor("00K81747629", "sorter")


    rentgen_check_camera = sorter_camera
    # rentgen_check_camera = cameras.CamProcessor("XXXXXXXXX")

    # sorter_camera.stream_vision()


    print(" I N I T    B A S E    D A T A  ")
    print("--------------------------------")

    # preprocess base image (printing template)
    dmc_code, inverted_dilated_base, base_data_dict_list, base_img, text_mask = process_base_image()


    print("  D O    C A L I B R A T I O N  ")
    print("--------------------------------")

    robot_points = np.array([])
    # try to load calib matrix, if it doesnt load any - perform calibration procedure
    if (not sorter_camera.load_calib_matrix()):
        input("put calibration board under camera. press enter when ready")
        # get chessboard points and image with them labeled in order
        img, camera_points = sorter_camera.get_calibration_points()

        d.show_image(img)

        # start drag mode
        robot_points = np.array(rob.get_draging_points(), dtype=np.float32)[:, :-1]

        # calculate and save transformation matrix
        sorter_camera.compute_calibration(camera_points,robot_points)


    """# fun part - left click to move robot to position on image
    image_mov = lambda coords: rob.mov(sorter_camera.imagePoint_to_robotPoint(coords), mtype="movj")
    d.set_click(image_mov)
    # TODO, image doesnt respond without cv2.waitKey()
    # there has to be loop waiting for clicks and/or loop break
    # this means manualy focus on image and press keys (or click)
    d.show_image(sorter_camera.get_image())
    # active wait (blocking), break pressing esc or n
    d.active_wait()
    """

    # TODO set positions atributes of robot
    # print("drag robot to these points and press enter:")
    # print("position above edge, position target (droping area) and into working height")
    # position_edge, position_target, working_z_height = rob.get_draging_points()[:3]
    
    # position_edge = list(position_edge)
    # position_target = list(position_target)
    # working_z_height = working_z_height[2]

    # CAREFUL - hard coded positions... uncoment lines right above this
    position_edge = [310, 170, 32 , -90]
    position_target =[307, 277, 34 , -90]
    working_z_height = -80.



    # position_0 = [8, 240, 0]  #[x,y,z]
    # position_target = [8, 240, -50 , 0]
    # working_z_height = 0
    # cameramid = cca [300, 9, -120, 0]
    rob.mov(position_edge)
    time.sleep(1)


    d.show_image(sorter_camera.get_image())


    input("clear the space in front of sorter camera and around robot! - press enter to confirm")
    sorter_camera.capture_background()
    print_check_camera.capture_background()
    d.show_image(sorter_camera.background)
    # cv2.destroyAllWindows()



    robot_points = robot_points.tolist()
    if len(robot_points) == 0:
        robot_points.append(sorter_camera.imagePoint_to_robotPoint([1500,1000]).tolist())
        # print("TUTOK:", robot_points[-1])

    # try and check all position moves for any mistakes and colision
    path = [position_edge] + [robot_points[0][:-1]+[working_z_height]] + robot_points + [robot_points[-1][:-1]+[working_z_height]] + [position_edge] + [position_target] + [position_edge]
    print("path: ",path)
    rob.move_path(path)



    print("S T A R T    M A I N    L O O P ")
    print("--------------------------------")

    def wait_for_checker_thread(pos):
        global checker_free
        global rob
        if(checker_free == None):
            checker_free = True
            rob.mov(pos)
            return

        while(not checker_free):
            time.sleep(0.1)
        # drop the package
        rob.drop_pack()
        checker_free = False
        # move above target and neutral rotation
        rob.mov(pos)

    def sorting_thread():
        global checker_free
        global image_queue
        global sorter_camera

        while(True):
            # start thread for checking checker_free and if it frees, drop the package
            # thread:
            waiter = threading.Thread(target=wait_for_checker_thread, args=(position_edge[:3]+[-90],))
            waiter.start()

            # look for package on table
            sort_img = sorter_camera.lookup_package()



            # recognize package and rotation
            position_img, pack_img_position, angle = sorter_camera.get_pack_position(sort_img)

            image_queue.put(position_img)
        
            # compute rotation position of the arm head
            rot1, rot2 = robots.transform_angle_for_arm(angle)

            # transform image coordinates to arm coordinates
            pack_position = list(sorter_camera.imagePoint_to_robotPoint(pack_img_position))

            # join the thread - wait till the package is dropped
            waiter.join()


            # change Z if needed
            pack_position[2] = pack_position[2]
            # add R rotation of picking (half the angle)
            pack_position.append(rot1)
            # add R rotation of droping (other half the angle)
            position_target[3] = rot2
            position_edge[3] = rot2
            
            print("pack position: ", pack_position)
            print("pack target: ", position_target)


            # move above pack (change z of target -> move)
            rob.mov(pack_position[:2] + [working_z_height] + [pack_position[-1]])
            # grab package (go down z until DI)
            rob.suck()
            # blocking function
            rob.move_until_DI(pack_position, 3, False)
            
            # move to transition point (next to edge so the package will not hit the edge)
            tmp = position_edge
            tmp[1] = position_edge[1] - 70
            rob.mov(tmp)

            rob.mov(position_edge)
            # move package to target (move to position_target, turn of DO)
            rob.mov(position_target)

            # check if print checker is ready for new package, if so - drop package
            # if not, look for new package to transfer and precalculate coordinates to efficiently use CPU
            if (checker_free):

                # drop the package
                rob.drop_pack()
                # move above target and neutral rotation
                rob.mov(position_edge[:3]+[-90])

    def print_check_thread():
        global checker_free
        global print_check_camera
        global image_queue
        while(True):
            # return
            # look for image, ignore moving(changing) vision and background
            img = print_check_camera.lookup_package()
            # img = print_check_camera.get_image()
            start_time = time.time()

            rentgen_img = print_check_camera.get_image()
            # rentgen_img = rentgen_check_camera.get_image()

            # d.show_image(img)
            image_queue.put(img)


            # enhance image of package (add contrast, filter, ...) - this function is crucial (commented because called in process_package)
            # train_img = preprocess_trainImg(img)
            # d.show_image(train_img)

            # compare img with base - this function is the largest and takes the most computation time
            result = process_package(dmc_code, inverted_dilated_base, base_data_dict_list, base_img, img, text_mask, rentgen_img)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Processing time for function was:", elapsed_time, "seconds")

            image_queue.put(result)
            # d.show_image(result)

            # TODO HERE I OPEN ONE OR OTHER HARDWARE EXIT FOR PACKAGE
            # if(result): open good
            # else: open bad

            # wait for background (blocking), then change var leting another pack in
            print_check_camera.wait_for_background()
            checker_free = True

    # Create threads
    # t1 = threading.Thread(target=sorting_thread, args=(sorter_camera,))

    t1 = threading.Thread(target=sorting_thread)
    t2 = threading.Thread(target=print_check_thread)

    # Start threads
    t1.start()
    t2.start()

    while(True):
        try:

            img = image_queue.get(timeout=60)
            d.show_image(img)
            time.sleep(0.2)

        except queue.Empty:
            time.sleep(0.2)

    # Wait for threads to finish (in this case, they run indefinitely)
    t1.join()
    t2.join()





# start_time = time.time()
# end_time = time.time()
# elapsed_time = end_time - start_time
# print("Processing time for function was:", elapsed_time, "seconds")


# process_base_image()


# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\python_slider\\astronaut.jpg')
# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\python_slider\\croped_base.jpg')
# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\python_slider\\EMOXICEL_FAM_Trium.jpg')
# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\print_images\\base\\EMOXICEL_FAM_Trium_correct.jpg')
# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\print_images\\base\\EMOXICEL_FAM_Trium_wrongDate.jpg')

# train_img_orig = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\image.jpg')
# train_img_orig = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\deformed_astronaut.jpg')
# train_img_orig = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\deformed_image.jpg')
# train_img_orig = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\image.jpg')

'''

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("click on coordinates x:",x," y:",y)
        if param['click_fun'] == cv2.destroyAllWindows:
            cv2.destroyAllWindows()
        else:
            param['click_fun']([x,y,0])

        # pyautogui.press('e')
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyWindow(param['window_name'])

def show_images(images):
    for i, img in enumerate(images):
        cv2.namedWindow(f'Image {i}', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(f'Image {i}', mouse_callback, {'window_name': f'Image {i}'})
        cv2.resizeWindow(f'Image {i}', 1600, 900)  # Set your desired size here
        cv2.imshow(f'Image {i}', img)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def create_display( name="Image", click_fun = lambda coords: None):
    
    cv2.destroyAllWindows()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    cv2.setMouseCallback(name, mouse_callback, {'window_name': name, 'click_fun': click_fun})
    
    # Move window to specified display (in this case to the left and down)
    cv2.moveWindow(name, -3000, 500)

    # Set window properties for full screen
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # cv2.resizeWindow(name, 1600, 900)  # Set your desired size here
    
    # cv2.imshow(name, image)

    # cv2.waitKey(1)
    # # cv2.destroyAllWindows()


def show_image(image, name="Image", click_fun = lambda coords: None):
    
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    # cv2.setMouseCallback(name, mouse_callback, {'window_name': name, 'click_fun': click_fun})
    
    # # Move window to specified display (in this case to the left and down)
    # cv2.moveWindow(name, -3000, 500)

    # # Set window properties for full screen
    # cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # # cv2.resizeWindow(name, 1600, 900)  # Set your desired size here
    
    cv2.imshow(name, image)

    cv2.waitKey(1)
    # cv2.destroyAllWindows()


def split_image(image, rows, cols):
    
    height, width = image.shape[:2]

    # Calculate the size of each part
    part_height = height // rows
    part_width = width // cols

    # Initialize a list to store the split parts
    split_parts = []

    # Split the image into parts
    for i in range(rows):
        for j in range(cols):
            # Calculate the starting and ending row and column indices for the current part
            start_row = i * part_height
            end_row = (i + 1) * part_height
            start_col = j * part_width
            end_col = (j + 1) * part_width

            # Extract the current part from the image
            part = image[start_row:end_row, start_col:end_col]

            # Append the current part to the list of split parts
            split_parts.append(part)

    return split_parts

def template_registration():

    # Read the base image (template) and the printed image
    base = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\croped_base.jpg', cv2.IMREAD_GRAYSCALE)
    printed_image = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\image.jpg', cv2.IMREAD_GRAYSCALE)

    print("showing base and photo")
    d.show_images([base,printed_image])


    # Perform template matching
    result = cv2.matchTemplate(printed_image, base, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc

    # Draw a rectangle around the matched region (optional)
    w, h = base.shape[::-1]  # Get the width and height of the template
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(printed_image, top_left, bottom_right, 0, 2)

    # Display the result
    show_image(printed_image, "result")

def printshit(var):
    for name, value in globals().items():
        if value is var:
            if isinstance(var, np.ndarray):
                first_item = var.flatten()[0]  # Get the first item regardless of shape
                print(name + ":  " + str(first_item) + " shape=" + str(var.shape))
            else:
                print(name + ":  " + str(value))

def registration2():
    # # Read images
    base_image = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\croped_base.jpg')
    img_image = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\deformed_image.jpg')
    # # img_image = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\image.jpg')
    
    # base_image = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\astronaut.jpg')
    # img_image = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\deformed_astronaut.jpg')
    


    # Convert images to grayscale
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_image, cv2.COLOR_BGR2GRAY)

    # img_gray = np.array(img_image, dtype=base_gray.dtype)
    # img_gray = cv2.imread(img_image)
    # img_gray = cv2.cvtColor(img_image, cv2.COLOR_BGR2GRAY)

    
    # Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(base_gray, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(base_gray, kp1, img_gray, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display matches
    show_image(img_matches, "matches")

    # Calculate transformation matrix
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp base image onto img image
    warped_base = cv2.warpPerspective(base_image, M, (img_image.shape[1], img_image.shape[0]))

    # Overlay images
    overlay = cv2.addWeighted(img_image, 0.5, warped_base, 0.5, 0)

    # Show overlay
    show_image(overlay, "Overlay")

def deform_img():

    # Load image
    image_path = 'C:\\Users\\Uzivatel\\Desktop\\python_slider\\image.jpg'
    image_rgb = io.imread(image_path)

    # Convert to grayscale
    image = color.rgb2gray(image_rgb)

    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # Adjust the amplitude to cover the entire image
    amplitude = rows / 32  # Adjust as needed

    # Add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 4 * np.pi, src.shape[0])) * amplitude
    dst_cols = src[:, 0]
    
    # Center the warped image
    dst_rows -= np.min(dst_rows) - (rows - np.max(dst_rows)) / 2

    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = rows
    out_cols = cols
    out = warp(image, tform, output_shape=(out_rows, out_cols))

    fig, ax = plt.subplots()
    ax.imshow(out, cmap='gray')
    ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    ax.axis((0, out_cols, out_rows, 0))
    # plt.show()
    out_uint8 = img_as_ubyte(out)
    io.imsave('deformed_image.jpg', out_uint8)
    return out_uint8

def hamming_brute_matching(des1, des2):

    # Create a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # take just N best
    N = 15
    matches = matches[:N]
    
    return matches

def get_features_orb(base_img, train_img):
    # Detect keypoints and descriptors using ORB
    # orb = cv2.ORB_create(scaleFactor=3, edgeThreshold=20, firstLevel=1, WTA_K=2)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(base_img, None)
    kp2, des2 = orb.detectAndCompute(train_img, None)

    # Match descriptors
    # matches = knn_matching(des1, des2)
    matches = hamming_brute_matching(des1, des2)
    # Extract corresponding keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    matched_img = cv2.drawMatches(base_img, kp1, train_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_image(matched_img)

    return src_pts, dst_pts

def get_features_surf(base_img, train_img):
    # Initialize SURF detector
    surf = cv2.xfeatures2d.SURF_create()

    # Detect keypoints and compute descriptors for base image
    kp1, des1 = surf.detectAndCompute(base_img, None)

    # Detect keypoints and compute descriptors for train image
    kp2, des2 = surf.detectAndCompute(train_img, None)


    matches = knn_brute_matching(des1, des2, k=2)

    # Extract corresponding keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matched_img = cv2.drawMatches(base_img, kp1, train_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_image(matched_img)


    return src_pts, dst_pts

def get_features_akaze(base_img, train_img):

    # Initialize AKAZE detector
    akaze = cv2.AKAZE_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = akaze.detectAndCompute(base_img, None)
    kp2, des2 = akaze.detectAndCompute(train_img, None)


    matches = knn_brute_matching(des1, des2)

    # Extract corresponding keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matched_img = cv2.drawMatches(base_img, kp1, train_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_image(matched_img)

    return src_pts, dst_pts

def get_features_brisk(base_img, train_img):

    # Initialize BRISK detector
    brisk = cv2.BRISK_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = brisk.detectAndCompute(base_img, None)
    kp2, des2 = brisk.detectAndCompute(train_img, None)

    # Create a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Extract corresponding keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matched_img = cv2.drawMatches(base_img, kp1, train_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_image(matched_img)

    return src_pts, dst_pts

def filter_outliers(src_pts, dst_pts, distance_threshold = 20, grid = (192,128)):
    
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # transformed points
    # trans_pts = np.dot(H, src_pts)
    trans_pts = cv2.perspectiveTransform(src_pts, H)

    # Calculate distances from the wanted points
    distances = np.linalg.norm(dst_pts - trans_pts, axis=2)
    # distances = np.linalg.norm(dst_pts - src_pts - avg_vector, axis=1)

    # Filter matches based on distance threshold
    # filtered_indices = [i for i, dist in enumerate(distances) if dist < distance_threshold]
    filtered_indices = np.where(distances < distance_threshold)[0]
    filtered_src_pts = src_pts[filtered_indices]
    filtered_dst_pts = dst_pts[filtered_indices]

    filtered_src_pts, filtered_dst_pts = grid_sampling(filtered_src_pts, filtered_dst_pts, grid)

    # return filtered_src_pts, filtered_dst_pts, filtered_indices
    return filtered_src_pts, filtered_dst_pts, H

def grid_sampling(src_pts, dst_pts, grid_size):
    """
    Perform grid-based sampling to reduce the number of keypoints.

    Args:
    - keypoints: Array of keypoints (N, 2)
    - grid_size: Size of the grid (e.g., (10, 10) for a 10x10 grid)

    Returns:
    - sampled_keypoints: Sampled keypoints after grid-based sampling
    """
    # Calculate grid indices for each keypoint
    grid_indices = src_pts / grid_size

    # Group keypoints by grid cell
    grid_src_pts = {}
    grid_dst_pts = {}
    for idx, src, dst in zip(grid_indices, src_pts, dst_pts):
        idx_tuple = tuple(idx[0].astype(int))
        # print("name:idx_tuple, type:", type(idx_tuple), " value:", idx_tuple)
        # print("name:idx, type:", type(idx[0].astype(int)), " value:", idx[0].astype(int))

        if idx_tuple not in grid_src_pts:
            grid_src_pts[idx_tuple] = []
            grid_dst_pts[idx_tuple] = []
        grid_src_pts[idx_tuple].append(src)
        grid_dst_pts[idx_tuple].append(dst)

    # Sample one keypoint per grid cell (e.g., using the centroid)
    sampled_src = [np.mean(points, axis=0) for points in grid_src_pts.values()]
    sampled_dst = [np.mean(points, axis=0) for points in grid_dst_pts.values()]

    return np.array(sampled_src), np.array(sampled_dst)

@time_measure
def registration_warp(base_img, train_img):

    # get points matched to each other
    src_pts, dst_pts = get_features_sift(base_img, train_img)

    # filter those which are way too off position
    filtered_src_pts, filtered_dst_pts, H = filter_outliers(src_pts, dst_pts, distance_threshold = 30, grid=(96,64))

    # add corner points
    
    # Get the shape of the base image
    base_img_height, base_img_width = base_img.shape[:2]

    # Define the corner points of the base image
    corner_points_base = np.array([[0, 0], [base_img_width - 1, 0], [base_img_width - 1, base_img_height - 1], [0, base_img_height - 1]], dtype=np.float32)

    # Reshape corner points for src_pts
    corner_points_base_src = corner_points_base.reshape(-1, 1, 2)

    # Compute the corresponding points after applying the homography H
    corner_points_transformed = cv2.perspectiveTransform(corner_points_base_src, H)

    # Append corner points to src_pts and transformed points to dst_pts
    filtered_src_pts = np.vstack((filtered_src_pts, corner_points_base_src))
    filtered_dst_pts = np.vstack((filtered_dst_pts, corner_points_transformed))

    print(filtered_src_pts.shape)

    # Estimate piecewise affine transformation
    tform = PiecewiseAffineTransform()
    tform.estimate(filtered_src_pts.squeeze(), filtered_dst_pts.squeeze())

    # Warp the base image
    # warped_base = warp(base_image, tform.inverse, output_shape=img_image.shape[:2])
    warped_base = warp(base_img.astype(np.float32), tform.inverse, output_shape=train_img.shape[:2]).astype(train_img.dtype)
    
    # add keypoints used for warping
    warped_base_display = cv2.cvtColor(warped_base.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for pt in filtered_dst_pts.squeeze():
        # Convert to tuple and round to integer
        pt_int = (int(round(pt[0])), int(round(pt[1])))
        # Draw circle at keypoint location
        cv2.circle(warped_base_display, pt_int, radius=6, color=(0, 0, 255), thickness=2)  # Red color for keypoints


    # Overlay images
    # overlay = cv2.addWeighted(train_img, 0.1, warped_base, 0.9, 0)
    # overlay = cv2.subtract(train_img, warped_base)

    # Show overlay
    # show_image(overlay,"overlay")
        

    return warped_base_display

@time_measure
def registration_homography(base_img, train_img):

    
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
    # show_image(overlay,"overlay")
    warped_base = cv2.cvtColor(warped_base.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return warped_base

def registration1():
    # Read images
    image = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\image.jpg')
    base = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\base.jpg')

    # Convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    _, image_gray = cv2.threshold(image_gray ,177,255, cv2.THRESH_BINARY)


    # Find keypoints and descriptors using ORB
    orb = cv2.ORB_create()
    keypoints_image, descriptors_image = orb.detectAndCompute(image_gray, None)
    keypoints_base, descriptors_base = orb.detectAndCompute(base_gray, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_image, descriptors_base)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    points_image = np.float32([keypoints_image[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    points_base = np.float32([keypoints_base[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Draw matched keypoints on images
    img_with_keypoints = cv2.drawKeypoints(image_gray, keypoints_image, None, color=(0, 0, 255), flags=0)
    base_with_keypoints = cv2.drawKeypoints(base_gray, keypoints_base, None, color=(0, 0, 255), flags=0)
    """ 
        # Show images with highlighted keypoints
        cv2.namedWindow('Image with KeyPoints', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image with KeyPoints', 1600, 900)  # Set your desired size here
        cv2.imshow('Image with KeyPoints', img_with_keypoints)

        cv2.namedWindow('Base with KeyPoints', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Base with KeyPoints', 1600, 900)  # Set your desired size here
        cv2.imshow('Base with KeyPoints', base_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find homography
    """
    H, mask = cv2.findHomography(points_base, points_image, cv2.RANSAC, 5.0)


    # Draw only inliers, indicated by mask
    inliers_image = [keypoints_image[i] for i in range(len(mask)) if mask[i] == 1]
    inliers_base = [keypoints_base[i] for i in range(len(mask)) if mask[i] == 1]


    img_with_all_keypoints = cv2.drawKeypoints(img_with_keypoints, inliers_image, None, color=(0, 255, 0), flags=0)
    base_with_all_keypoints = cv2.drawKeypoints(base_with_keypoints, inliers_base, None, color=(0, 255, 0), flags=0)

    cv2.namedWindow('Image with KeyPoints', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image with KeyPoints', 1600, 900)  # Set your desired size here
    cv2.imshow('Image with KeyPoints', img_with_all_keypoints)

    cv2.namedWindow('Base with KeyPoints', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Base with KeyPoints', 1600, 900)  # Set your desired size here
    cv2.imshow('Base with KeyPoints', base_with_all_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # Warp base image using homography
    registered_base = cv2.warpPerspective(base_gray, H, (image.shape[1], image.shape[0]))

    # Save registered image
    cv2.imwrite('C:\\Users\\Uzivatel\\Desktop\\python_slider\\registered_base.jpg', registered_base)

    # Resize window
    cv2.namedWindow('Registered Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Registered Image', 1600, 900)  # Set your desired size here

    # Display registered image
    cv2.imshow('Registered Image', registered_base)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Overlay images for visual inspection
    overlay = cv2.addWeighted(image_gray, 0.5, registered_base, 0.5, 0)
    cv2.namedWindow('Overlay Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Overlay Image', 1600, 900)  # Set your desired size here

    cv2.imshow('Overlay Image', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''