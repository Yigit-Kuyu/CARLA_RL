import carla
import random,time


# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
#client = carla.Client('10.0.169.14', 2000)
world = client.get_world()
client.load_world('Town04')

# Spawn vehicle

blueprint_library = world.get_blueprint_library()

# Get the map's spawn points
spawn_point = random.choice(world.get_map().get_spawn_points())
# Get the blueprint library and filter for the vehicle blueprints
vehicle_blueprint = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
ego_vehicle = world.spawn_actor(vehicle_blueprint,spawn_point)


# Spawn a segmentation camera on the vehicle
cam_init_spawn_point = carla.Transform()
seg_cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
seg_camera = world.spawn_actor( seg_cam_bp, cam_init_spawn_point, attach_to=ego_vehicle )

# Saving images from sensor to disk:
seg_camera .listen(lambda image: image.save_to_disk('output/image%d.png' % image.frame))



#Enable the auto=pilot of the vehicle
ego_vehicle.set_autopilot(True)


#  Wait some time until the camera gathers enough data (e.g. 1 minutes)
time.sleep(60)


# Arama: blueprint_library