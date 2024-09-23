# zapojím vše do elektriky
# zkontroluji že kamera svítí modře
# zapnu světlo
# spustím program:
# zeptám se na jméno datasetu
# vytvořím složku resources/jméno
# připojím ke kamerám
# ZAČÁTEK CYKLU
# začnu čekat na balík
# když přijde balík vyfotím oběma kamerama
# uložím do resources/jméno/kamera
# otevřu přepážku a čekám na to až odjede balík 
# zavřu přepážku


# POZNÁMKY K CHODU PROGRAMU:
# Bedna má obrovský problém se statickou elekřinou, balíky se extrémně často přilepí ke sklu a nekloužou
# tento problém otestoval schopnost programu poznat že balík neodjel ze skluzu (funguje)
# Bylo otestování posílání balíků vzhůru nohama (po skle klouzala papírová část) - zde nebyl problém s přilepením
# Balíky focené vzhůru nohama mají 
#   výhody:
    # nelepí se na sklo
    # svou vahou se srovná povrch (není deformace sváru a rohů)
#   nevýhody:
    # při průhledu rentgenem je čitelný natištěný text a tedy může překážet v dalším zpracování
    # při případném vyměnění směru nasvícení (kamer) by se odráželo světlo od skla do kamery (rentgen by fungoval lépe, ale nebylo by možné vyfotit balík z druhé strany (odlesky))

# je znatelný stín na kraji bedny u přepážky, teoreticky by neměl vadit a mohlo by ho jít automaticky staticky odstranit (lokálně zvýšit jas)


import cv2
import os
import cameraControl.cam_processor as cameras
import serial
import time
import displayControl.display as displays
import keyboard


# ask for dataset name and label
dataset_name = "2_large_secondary"
#dataset_name = input("Enter dataset name: ")
label_cam1 = "print_checker"
label_cam2 = "rentgen"

# create a folder with the name in /resources/name/label
path_print = os.path.join("resources", dataset_name, label_cam1)
os.makedirs(path_print, exist_ok=True)
path_rentgen = os.path.join("resources", dataset_name, label_cam2)
os.makedirs(path_rentgen, exist_ok=True)


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

time.sleep(0.1)
#input("Press enter to capture the background")
camera_print.capture_background()
d.show_image(camera_print.background)


r=1
# wait for the user to press enter to take a picture
while True:
    # w = input("Press enter to take a picture or type exit to exit:")
    # if w == 'exit':
    #     break
    if keyboard.is_pressed('p'):
        print("Pausing")
        input("Press enter to continue")
        print("Continuing")

    frame = camera_print.lookup_package()
    frame = camera_print.get_rgb_image()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(path_print, f"{len(os.listdir(path_print))}.png"), frame)
    frame_rentgen = camera_rentgen.get_rgb_image()
    frame_rentgen = cv2.cvtColor(frame_rentgen, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(path_rentgen, f"{len(os.listdir(path_rentgen))}.png"), frame_rentgen)

    d.show_image(frame)

    # open the gate
    print("Opening the gate")
    if(r==1):
        open_gate_ok()
        time.sleep(1)
        #r=0
    else:
        open_gate_fail()
        r=1

    if keyboard.is_pressed('p'):
        print("Pausing")
        input("Press enter to continue")
        print("Continuing")

    # wait till package is gone
    print("Waiting for the package to leave")
    try:
        after_leav_img = camera_print.wait_package_leave()
        # d.show_image(after_leav_img)
        # d.active_wait()
    # except when value error is raised
    except ValueError as e:
        print("Error: ", e)
        input("Press enter after you cleared the slide from mess to continue")
        close_gate()
        time.sleep(0.2)
        camera_print.update_background()
        continue
    

    close_gate()



