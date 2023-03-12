
from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import random
import time
import numpy as np
import cv2 # import OpenCV


############################## Part_1
actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0) # 2 saniye yanıt süresi verdim

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    # Pick Car:

    # filter tesla model 3 vehicle, get blueprint of vehicle
    bp_tesla = blueprint_library.filter('model3')[0] # We want the item at zero index

    # Carla comes with something like 200 spawn points, so we can just pick one of those randomly:
    spawn_point = random.choice(world.get_map().get_spawn_points()) # Spawn vehicle somewhere (randomly)
    # Spawn the ego vehicle
    vehicle = world.spawn_actor(bp_tesla, spawn_point)

    #  When set_autopilot is True, the Traffic Manager passed as parameter will move the vehicle around.
    # vehicle.set_autopilot(True)


############################## Part_2
    # Note: Everytime, we can spawn anything


    # control the car
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0)) # just go straight
    # Not forget to add this vehicle to our list of actors that we need to track and clean up:
    actor_list.append(vehicle)





############################## Part_3
    # Actually, we need to a couple of other things,
    # one we need a camera, two we need a collision sensor
    # I want to attach camera sensor


    # blueprint for rgb camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')

    # If you want to feed NN, you should be very specific in camera attributes.
    #  Modify the attributes of the blueprint to set image resolution (the dimensions of the image, x-y)
    camera_bp.set_attribute('image_size_x', '1920') # x=1920
    camera_bp.set_attribute('image_size_y', '1080') # y=1080
    # Modify the attributes of the blueprint to set the field of view
    camera_bp.set_attribute('fov', '110') # fov in degrees, kaydedidilecek imageleri direkt etkileyecek
    # Set the time in seconds between sensor captures
    camera_bp.set_attribute('sensor_tick', '1.0')

    # Now,  we need to add this camera to our car.
    # Adjust the sensor from a relative location to the vehicle
    spawn_point_camera=carla.Transform(carla.Location(x=2.5, z=0.7)) # move forward 2.5 and up 0.7 (negative is down)

    # spawn the camera and attach it to our ego vehicle
    sensor = world.spawn_actor(camera_bp, spawn_point_camera, attach_to=vehicle)
    # add sensor to list of actors
    actor_list.append(sensor)

    # Saving images from sensor to disk by using built-in function:
    sensor.listen(lambda image: image.save_to_disk('output/image%d.png' % image.frame))


############################## Part_4
    # We would like to modify lamda function

    def process_img(image):
        IM_HEIGHT=480
        IM_WIDTH=640
        i = np.array(image.raw_data)  # convert to an array
        print(i.shape)
        # was flattened, so we are going to shape it:
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # image=widthXheight, 4 for RGB + Alpha (By the way, I don't care about alpha value may be later)
        # get the first three element (no alpha):
        i3 = i2[:, :, :3]  # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB, we want RGB values)
        cv2.imshow("title", i3)  # show it. (openCV method)
        cv2.waitKey(1) # one milisecond wait
        # We are working imaginary data passing through NN that prefers information generally to 0 or 1, -1 or 1.
        return i3 / 255.0  # Values 0-255 are silly to NN, so we "normalize(scale)" them to be between 0-1, we will actually get a representation of what the camera sees


    # Saving images from sensor to disk by using created function (do something with this sensor):
    sensor.listen(lambda data: process_img(data))



    time.sleep(5)  # Before destroy the spawned vehicle, wait for 5 seconds
finally:

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')




