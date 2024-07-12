import numpy as np
import time
import threading
import queue
import keyboard
# import sys

# from tools import *
from threadControl.threadControl import ThreadControl
import cameraControl.cam_processor as cameras
from imageProcessing.imageTools import process_base_image
import robotControl.robot_processor as robots
import displayControl.display as displays
from threadControl.threadFunctions import dummy_print_check_thread, print_check_thread, rentgen, sorting_thread




# variable for display
d = None

# robot var
rob = None


def on_key_event(event, shared):
    if(event.name == 'c'):
        shared.exit_event.set()
        if shared.exit_event.is_set():
            print("exit event set")
        rob.stop_move()
        rob.disable()
        global d
        del d
        # global exit_event
    print("pressed: ", event.name)


if __name__ == '__main__':

    print("      I N I T    R O B O T      ")
    print("--------------------------------")

    # rob = robots.RobotProcessor()
    rob = robots.DumbRobotProcessor()

    print("    I N I T    C A M E R A S    ")
    print("--------------------------------")


    d = displays.DisplayProcessor()

    # list cameras info
    cameras.list_devices()

    # connect to cam by serial number of camera and create that camera object
    # DA1274571
    # DA3215511
    # 00K81747629


    print_check_camera = cameras.CamProcessor("DA1274571", "print_checker")
    # print_check_camera.stream_vision(d)
    # sorter_camera = cameras.CamProcessor("00K81747629", "sorter")
    # sorter_camera.stream_vision(d)

    rentgen_check_camera = cameras.CamProcessor("DA3215511", "rentgen")
    # rentgen_check_camera.stream_vision(d)


    sorter_camera = print_check_camera


    print(" I N I T    B A S E    D A T A  ")
    print("--------------------------------")

    # preprocess base image (printing template)
    base_info = process_base_image()


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

    # CAREFUL - hard coded positions... uncomment lines right above this
    position_edge = [310, 170, 32 , -90]
    position_target =[307, 277, 34 , -90]
    working_z_height = -80.


    rob.mov(position_edge)
    time.sleep(1)



    # input("clear the space in front of sorter camera and around robot! - press enter to confirm")
    sorter_camera.capture_background()
    rentgen_check_camera.capture_background()
    print_check_camera.capture_background()
    d.show_image(sorter_camera.background)
    # cv2.destroyAllWindows()



    robot_points = robot_points.tolist()
    if len(robot_points) == 0:
        robot_points.append(sorter_camera.imagePoint_to_robotPoint([1500,1000]).tolist())

    # try and check all position moves for any mistakes and colision
    path = [position_edge] + [robot_points[0][:-1]+[working_z_height]] + robot_points + [robot_points[-1][:-1]+[working_z_height]] + [position_edge] + [position_target] + [position_edge]
    print("path: ",path)
    rob.move_path(path)



    # load image from file
    import cv2
    img = cv2.imread('.\\resources\\general\\svar_pattern19x19.png')
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # print img as array with commas
    for row in img.tolist():
        print(row,",")
    d.show_image(img)
    # d.active_wait()



    print("S T A R T    M A I N    L O O P ")
    print("--------------------------------")


    shared = ThreadControl(rob)

    # Create threads
    t1 = threading.Thread(target=sorting_thread, daemon=True, args=(sorter_camera, shared, position_edge, position_target, working_z_height))
    t2 = threading.Thread(target=print_check_thread, daemon=True, args=(print_check_camera, rentgen_check_camera, shared, base_info))
    
    # dummy thread for testing
    # t2 = threading.Thread(target=rentgen, daemon=True, args=(print_check_camera, shared, base_info))
    # t2 = threading.Thread(target=dummy_print_check_thread, daemon=True, args=(print_check_camera, shared, base_info))



    # Start threads
    # t1.start()
    t2.start()


    keyboard.hook(lambda event: on_key_event(event, shared))
    # keyboard.hook(on_key_event())


    while(True):
        try:

            img = shared.image_queue.get(block=False)

            d.show_image(img)

            # save img to file
            import cv2
            cv2.imwrite('C:\\Users\\Uzivatel\\Desktop\\python_slider\\image.jpg', img)

            time.sleep(0.2)

        except queue.Empty:
            if shared.exit_event.is_set():
                print("exit event")
                exit()
            # print("no image in queue")
            time.sleep(0.2)



    # Wait for threads to finish (in this case, they run indefinitely)
    t1.join()
    t2.join()












# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\python_slider\\astronaut.jpg')
# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\python_slider\\croped_base.jpg')
# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\python_slider\\EMOXICEL_FAM_Trium.jpg')
# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\print_images\\base\\EMOXICEL_FAM_Trium_correct.jpg')
# base = load_baseImg('C:\\Users\\Uzivatel\\Desktop\\print_images\\base\\EMOXICEL_FAM_Trium_wrongDate.jpg')

# train_img_orig = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\image.jpg')
# train_img_orig = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\deformed_astronaut.jpg')
# train_img_orig = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\deformed_image.jpg')
# train_img_orig = cv2.imread('C:\\Users\\Uzivatel\\Desktop\\python_slider\\image.jpg')