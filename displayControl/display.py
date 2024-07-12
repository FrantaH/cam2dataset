import cv2
import numpy as np
import time
import pyautogui
import matplotlib.pyplot as plt
import os



# ANSI escape codes for text colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

class DisplayProcessor:

    def __init__(self, name = "Image", click_fun = lambda coords: None):
        '''
        inicialize display object for showing images in one window
        '''
        self.name = name
        cv2.destroyAllWindows()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        # cv2.setMouseCallback(name, self.mouse_callback, {'window_name': name, 'click_fun': click_fun})

        # Move window to specified display (in this case to the left and down)
        # cv2.moveWindow(name, -3000, 500)
        cv2.moveWindow(name, 0, 0)



        # # cv2.resizeWindow(name, 1600, 900)  # Set your desired size here

        # Set window properties for full screen
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # print(os.path.dirname(__file__))

        # optionaly
        black_image = np.zeros((500, 500), dtype=np.uint8)
        self.show_image(black_image)

    def __del__(self):
        print("deleting display object")
        cv2.destroyWindow(self.name)

    def reset(self):
        self.__init__(self.name)

    def set_click(self, click_fun):
        cv2.setMouseCallback(self.name, self.mouse_callback, {'window_name': self.name, 'click_fun': click_fun})

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("click on coordinates x:",x," y:",y)
            print("collor:", self.image[y,x])
            # print(param)
            if param['click_fun'] == cv2.destroyAllWindows:
                cv2.destroyAllWindows()
            else:
                param['click_fun']([x,y,0])

            # pyautogui.press('e')
        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.destroyWindow(param['window_name'])

    def show_images(self, images):
        for i, img in enumerate(images):
            cv2.namedWindow(f'Image {i}', cv2.WINDOW_NORMAL)
            # cv2.setMouseCallback(f'Image {i}', mouse_callback, {'window_name': f'Image {i}'})
            cv2.resizeWindow(f'Image {i}', 1600, 900)  # Set your desired size here
            cv2.imshow(f'Image {i}', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_image(self, image):
        
        # cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        # cv2.setMouseCallback(name, mouse_callback, {'window_name': name, 'click_fun': click_fun})

        # # Move window to specified display (in this case to the left and down)
        # cv2.moveWindow(name, -3000, 500)

        # # Set window properties for full screen
        # cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.image = image
        cv2.imshow(self.name, image)

        cv2.waitKey(10)
        # cv2.destroyAllWindows()

    def active_wait(self):
        while True:
            print("blocking display")
            print("end by pressing escape or 'n'")
            key = cv2.waitKey(0) & 0xFF
            print("key pressed, code: ", str(key))
            if "\27" == key or ord('n') == key:
                break


if __name__ == '__main__':

    # list cameras info
    # cameras.list_devices()

    # sorter_camera = cameras.CamProcessor("00K81747629")

    # sorter_camera.stream_vision()

    # sorter_camera.capture_background()

    # connect to cam by serial number of camera and create that camera object
    # DA1274571
    # 00K81747629

    # print_check_camera = cameras.CamProcessor("DA1274571")

    # sorter_camera = cameras.CamProcessor("00K81747629")

    # rentgen_check_camera = sorter_camera
    # # rentgen_check_camera = cameras.CamProcessor("XXXXXXXXX")
    import os
    display = DisplayProcessor()
    print(os.path.abspath("."))

    time.sleep(1)

    del display