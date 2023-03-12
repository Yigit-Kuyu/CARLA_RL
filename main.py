import carla
import random


# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
#client = carla.Client('10.0.169.14', 2000)
world = client.get_world()
client.load_world('Town04')
####### Spectator navigation
'''
# Retrieve the spectator object
spectator = world.get_spectator()

# Get the location and rotation of the spectator through its transform
transform = spectator.get_transform()

location = transform.location
rotation = transform.rotation

# Set the spectator with an empty transform
spectator.set_transform(carla.Transform())
# This will set the spectator at the origin of the map, with 0 degrees
# pitch, yaw and roll - a good way to orient yourself in the map

'''

####################


# Get the blueprint library and filter for the vehicle blueprints
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn 50 vehicles randomly distributed throughout the map
# for each spawn point, we choose a random vehicle from the blueprint library
for i in range(0,50):
    r_c_b=random.choice(vehicle_blueprints)
    r_c_s=random.choice(spawn_points)
    world.try_spawn_actor(r_c_b,r_c_s)


c_b=random.choice(vehicle_blueprints)
c_s= random.choice(spawn_points)
ego_vehicle = world.spawn_actor(c_b,c_s)


########### Add sensors
'''
# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Start camera with PyGame callback
#camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))
'''

###################


# For RGB Camera
# Find the blueprint of the sensor.
blueprint_camera = world.get_blueprint_library().find('sensor.camera.rgb')
# Modify the attributes of the blueprint to set image resolution and field of view.
blueprint_camera.set_attribute('image_size_x', '1920')
blueprint_camera.set_attribute('image_size_y', '1080')
blueprint_camera.set_attribute('fov', '110')
# Set the time in seconds between sensor captures
blueprint_camera.set_attribute('sensor_tick', '1.0')

# For Lidar
# Find the blueprint of the sensor.
blueprint_lidar = world.get_blueprint_library().find('sensor.lidar.ray_cast')
blueprint_lidar.set_attribute('range', '50')


# Spawning:
transform = carla.Transform(carla.Location(x=0.8, z=1.7))  # When spawning with attachment, location must be relative to the parent actor.
sensor = world.spawn_actor(blueprint_camera, transform, attach_to=ego_vehicle)




