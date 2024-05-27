import threading
import time

import threadControl
import robotControl.robot_processor as robots
from imageProcessing.imageTools import process_package



def wait_for_checker_thread(pos, shared):
        
        if(not shared.robot.sucking):
            shared.checker_free.set()
            shared.robot.mov(pos)
            return

        shared.checker_free.wait()
        shared.checker_free.clear()
        # drop the package
        shared.robot.drop_pack()
        # move above target and neutral rotation
        shared.robot.mov(pos)

def sorting_thread(camera, shared, position_edge, position_target, working_z_height):
    
    rob = shared.robot
    while(True):
        # start thread for checking checker_free and if it frees, drop the package
        # thread:
        waiter = threading.Thread(target=wait_for_checker_thread, daemon=True, args=(position_edge[:3]+[-50],))
        waiter.start()

        # look for package on table
        sort_img = camera.lookup_package()



        # recognize package and rotation
        position_img, pack_img_position, angle = camera.get_pack_position(sort_img)

        shared.image_queue.put(position_img)
    
        # compute rotation position of the arm head
        rot1, rot2 = robots.transform_angle_for_arm(angle)

        # transform image coordinates to arm coordinates
        pack_position = list(camera.imagePoint_to_robotPoint(pack_img_position))

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
        tmp = position_edge[:]
        tmp[1] = position_edge[1] - 70
        rob.mov(tmp)

        rob.mov(position_edge)
        # move package to target (move to position_target, turn of DO)
        rob.mov(position_target)

        # check if print checker is ready for new package, if so - drop package
        # if not, look for new package to transfer and precalculate coordinates to efficiently use CPU
        if (shared.checker_free.is_set()):

            # drop the package
            rob.drop_pack()
            # move above closer to picking area but away from camera
            rob.mov(position_edge[:3]+[-50])

def print_check_thread(camera, shared, baseinfo):
    
    while(True):
        # return
        # look for image, ignore moving(changing) vision and background
        img = camera.lookup_package(threshold=3)
        # img = print_check_camera.get_image()
        start_time = time.time()

        rentgen_img = camera.get_image()
        # rentgen_img = rentgen_check_camera.get_image()

        # d.show_image(img)
        shared.image_queue.put(img)


        # enhance image of package (add contrast, filter, ...) - this function is crucial (commented because called in process_package)
        # train_img = preprocess_trainImg(img)
        # d.show_image(train_img)

        # compare img with base - this function is the largest and takes the most computation time
        result = process_package(baseinfo, img, rentgen_img)
        # result = process_package(dmc_code, inverted_dilated_base, base_data_dict_list, base_img, img, text_mask, rentgen_img)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Processing time for function was:", elapsed_time, "seconds")

        shared.image_queue.put(result)
        # d.show_image(result)

        # TODO HERE I OPEN ONE OR OTHER HARDWARE EXIT FOR PACKAGE
        # if(result): open good
        # else: open bad

        # wait for background (blocking), then change var leting another pack in
        camera.wait_for_background()
        shared.checker_free.set()

def dummy_print_check_thread(camera, shared, _):
    while(True):
        
        # img = print_check_camera.get_image()
        time.sleep(4)
        # image_queue.put(img)
        shared.checker_free.set()

