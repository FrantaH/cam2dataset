import cv2
import numpy as np
import time
import robotControl.robot_processor as robots

print("fuck")


# ANSI escape codes for text colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"




if __name__ == '__main__':

    print("      I N I T    R O B O T      ")
    print("--------------------------------")

    rob = robots.RobotProcessor()

    rob.reset()
    rob.unset_DO(list(range(1,17)))
    rob.disable()
