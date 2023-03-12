
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
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.models import Model

############################## Part_1
#We are gonno be talking about doing reinforcement learning 8 specifically,
# Deep-Q Learning with KARLA environment.
# They have kind of set standards for how to approach and work with environments to do RL

SHOW_PREVIEW = False # We dont want to see previews for less use of the computer resources
SECONDS_PER_EPISODE= 10 # ten seconds
IMG_WIDTH=640
IMG_HEIGHT=480
REPLAY_MEMORY_SIZE=5000
MIN_REPLAY_MEMORY_SIZE=1000
MINIBATCH_SIZE=16
PREDICTION_BATCH_SIZE=1
TRAINING_BATCH_SIZE=MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY=5
MODEL_NAME= 'Xception'

MEMORY_FRACTION=0.8
MIN_REWARD=-200
DISCOUNT=0.99
EPISODES=100 #How many episodes we want to do
AGGREGATE_STATS_EVERY=10

epsilon=1
Epsilon_decay=0.95



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


########## RL Structure for CARLA- Begin

class DQLAgent: # Deep Q Learning (Reinforcement learning algorithm)
    # Related RL tutorial: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
    def _init_(self):
        # We have a main network, which is constantly evolving, and then the target network, which we update every n things,
        # where n is whatever you want and things is something like steps or episodes:
        self.model=self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # As we train, we train from randomly selected data from our replay memory:
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) # REPLAY_MEMORY_SIZE: Memory of previous actions
        # For the same reasons as before (the RL tutorial), we will be modifying TensorBoard:
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        # we can wrap up our init method by setting some final values:
        self.target_update_counter = 0  # will track when it's time to update the target model
        self.graph = tf.get_default_graph()

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0  # To help us keeping episodes
        self.training_initialized = False  # waiting for TF to get rolling

        def create_model(self):
            # The first predictions/fitments when a model begins take extra long, so we're going to just pass some nonsense information initially to prime our model to actually get moving:
            base_model = Xception(weights=None, include_top=False, input_shape = (IM_HEIGHT, IM_WIDTH, 3)) # Xception Model: pre-trained model on the ImageNet database with python and Keras deep learning library.
            # 3: because we have  3 options-> left,straight, right

            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            predictions = Dense(3, activation="linear")(x) # Dense layer has 3 neurons,   because we have  3 options(posible predictions)-> left,straight, right
            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
            return model

        def update_replay_memory(self, transition): # for updating replay memory of DQLAgent
            # transition =  all of the information we need to train the model (current_state, action, reward, new_state, done)--> Tuple
            self.replay_memory.append(transition)

        # We only want to train if we have a bare minimum of samples in replay memory:
        def train(self): # Ensure we have enough replay memory to train
            # If we don't have enough samples, we'll just return:
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return
            # If we have enough samples, we will begin our training. First, we need to grab a random minibatch:
            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) # generate random minibatch
            # Once we have our minibatch, we want to grab our current and future q values:
            current_states = np.array([transition[0] for transition in minibatch]) / 255 # 255 for scaling
             # transition[0]: current_state, transition = (current_state, action, reward, new_state, done)
            with self.graph.as_default():
                current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

            new_current_states = np.array([transition[3] for transition in minibatch]) / 255
            with self.graph.as_default():
                future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

            # Create our inputs (X) and outputs (y):
            X = []
            y = []

            for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + DISCOUNT * max_future_q
                else: # if done
                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                X.append(current_state)
                y.append(current_qs)

            # We're only trying to log per episode, not actual training step, so we're going to use to keep track:
            log_this_step = False
            if self.tensorboard.step > self.last_logged_episode:
                log_this_step = True
                self.last_log_episode = self.tensorboard.step

            # We will fit:
            with self.graph.as_default():
                self.model.fit(np.array(X) / 255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                               callbacks=[self.tensorboard] if log_this_step else None)
            # callbacks will be "self.tensorboard" if log_this_step is True, otherwise there is no callbacks

            if log_this_step: # tracking for logging
                self.target_update_counter += 1

            # Check to see if it's time to update our target_model:
            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

            # We need a method to get q values (basically to make a prediction):
            def get_qs(self, state):
                return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

            # Finally, we just need to actually do training:
            def train_in_loop(self):
                X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
                y = np.random.uniform(size=(1, 3)).astype(np.float32)
                with self.graph.as_default():
                    self.model.fit(X, y, verbose=False, batch_size=1)

                self.training_initialized = True
                # To start, we use some random data like above to initialize, then we begin our infinite loop:
                while True:
                    if self.terminate: # Completed
                        return
                    self.train()
                    time.sleep(0.01)


########## RL Structure for CARLA- End



