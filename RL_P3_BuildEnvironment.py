
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
import math

print(cv2.__version__)

############################## Part_1
#We are gonno be talking about doing reinforcement learning 8 specifically,
# Deep-Q Learning with KARLA environment.
# They have kind of set standards for how to approach and work with environments to do RL

SHOW_PREVIEW = False # We dont want to see previews for less use of the computer resources
SECONDS_PER_EPISODE= 10 # ten seconds
IMG_WIDTH=640
IMG_HEIGHT=480

######### Environment Definition- Begin


class CarEnv:
     # Initial values:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0 # Steer amounts, it will be zero or one or two, or their negatives
                    # This means stter fully left, go straight or fully right


    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT
    actor_list = []

    front_camera = None
    collision_hist = []
    def __init__(self):
     self.client = carla.Client('localhost', 2000)  # connect server
     self.client.set_timeout(2.0) # two seconds

     # Once we have a client we can retrieve the world that is currently running.
     self.world = self.client.get_world()

     # The world contains the list blueprints that we can use for adding new actors into the simulation.
     blueprint_library = self.world.get_blueprint_library()

     # Now let's filter all the blueprints of type 'vehicle' and choose one at random.
     # print(blueprint_library.filter('vehicle'))
     # get the vehicle
     self.model_3 = blueprint_library.filter('model3')[0]

     def process_img(image): # for Lambda function to save image
         i = np.array(image.raw_data)  # convert to an array
          #print(i.shape)
         # np.save("iout.npy", i)
         # was flattened, so we are going to shape it:
         i2 = i.reshape((self.im_height, self.im_width, 4))  # image=widthXheight, 4 for RGB + Alpha (By the way, I don't care about alpha value may be later)
         # get the first three element (no alpha):
         i3 = i2[:, :, :3]  # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB, we want RGB values)
         if self.SHOW_CAM: # show the camera
             cv2.imshow("title", i3)  # show it. (openCV method)
             cv2.waitKey(1)  # one milisecond wait
         self.front_camera=i3



     def collision_data(self, event): # if we have an event, we will just append it to the hıstory
         self.collision_hist.append(event)

     def step(self, action): # take specific action
         '''
         For now let's just pass steer left, center, right?
         0, 1, 2
         '''
         if action == 0:
             self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0)) # throttle=1.0-->means full throttle, will go straith
         if action == 1:
             self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT)) # will go left
         if action == 2:
             self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT)) # will go right

         v = self.vehicle.get_velocity() # velocity
         kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)) # km per hour

         if len(self.collision_hist) != 0: # For registered any collision
             done = True
             reward = -200 # penalty
         elif kmh < 50: # Everthing is good, 50 kmh speed is threshold
             done = False
             reward = 0 # no penalty
         else:
             done = False
             reward = 1

         if self.episode_start + SECONDS_PER_EPISODE < time.time():
             done = True

         return self.front_camera, reward, done, None



     def reset(self):

        self.collision_hist = []
        self.actor_list = []

        # get vehicle
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # get RGB-Alpha camera
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}') # width
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}') # height
        self.rgb_cam.set_attribute('fov', '110') # field of view
        # attach RGB Camera
        transform = carla.Transform(carla.Location(x=2.5, z=0.7)) # Define the location of the camera
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle) # attach sensor to the vehicle
        self.actor_list.append(self.sensor) # add ıt to actor list
        self.sensor.listen(lambda data: self.process_img(data)) # save the frame with written function

        # if you dont do anything, if you just send ın the command to control, it makes the car react more quickly:
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)  # sleep for 4 seconds, (sleep to get things started and to not detect a collision when the car spawns/falls from sky.)

        # get collision sensor
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle) # add it to the same location of RGB
        self.actor_list.append(self.colsensor) # add ıt to actor list
        self.colsensor.listen(lambda event: self.collision_data(event)) # save the frame with written function

        # To effectively run "collision_data(event)", basically, it has to return observation well,
        # our observation is the front-facing camera (self.front_camera), so if after 4 seconds that camera still not ready,
        # we are going to say; if the camera returns None, wait:
        while self.front_camera is None: # Probably, we can do it for whole sensors
            time.sleep(0.01)

        # After we have built the process above, the episode has started:
        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.front_camera


######### Environment Definition- End






