# ask for dataset name and label
# create a folder with the name in /resources/name/label
# open and connect to the camera
# wait for the user to press enter to take a picture
# save the picture in the folder
# repeat until the user types 'exit'

import cv2
import os
import cameraControl.cam_processor as cameras

# ask for dataset name and label
dataset_name = input("Enter dataset name: ")
label = input("Enter label: ")

# create a folder with the name in /resources/name/label
path = os.path.join("resources", dataset_name, label)
os.makedirs(path, exist_ok=True)

# add the folders to gitignore if they are not already there
with open(".gitignore", "r") as f:
    path_ = os.path.join("resources", dataset_name)

    lines = f.readlines()
    if path_ not in lines:
        with open(".gitignore", "a") as f:
            f.write("\n" + path_)

# open and connect to the camera
camera = cameras.CamProcessor("DA1274571", "print_checker")
# camera = cameras.CamProcessor("DA3215511", "rentgen")

# wait for the user to press enter to take a picture
while True:
    w = input("Press enter to take a picture or type exit to exit:")
    if w == 'exit':
        break

    frame = camera.get_image()
    cv2.imwrite(os.path.join(path, f"{len(os.listdir(path))}.png"), frame)
    print("Picture saved")


