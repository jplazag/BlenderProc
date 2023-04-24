import blenderproc as bproc
import os
import numpy as np
import argparse
from blenderproc.python.utility.CollisionUtility import CollisionUtility
from typing import Dict
import mathutils
import glob
# import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument('treed_obj_path', help="Path to the downloaded 3D object")
parser.add_argument('output_dir', nargs='?', default="examples/automate_semantic_relations/Test_01/output", help="Path to where the final files, will be saved")
args = parser.parse_args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder) or not os.path.exists(args.treed_obj_path):
    raise OSError("One of the three folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

# load the front 3D objects
room_objs = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)

# Cache to fasten data Collection
bvh_cache : Dict[str, mathutils.bvhtree.BVHTree] = {}


def sample_pose(obj: bproc.types.MeshObject):
    # Sample the spheres location above the surface
    obj.set_location(bproc.sampler.upper_region(
        objects_to_sample_on=[surface_obj],
        min_height=1,
        max_height=4,
        use_ray_trace_check=False
    ))
    #Randomized rotation of the sampled object
    obj.set_rotation_euler(bproc.sampler.uniformSO3(around_y=False, around_x=False, around_z=True))






# define the camera intrinsics
bproc.camera.set_resolution(512, 512)

# Select the objects, where other objects should be sampled on
sample_surface_objects = []
for obj in room_objs:
    print(obj.get_name())
    if "table" in obj.get_name().lower() or "desk" in obj.get_name().lower():
        sample_surface_objects.append(obj)



# objects_of_interest = [ "op_microwave","red_mug"]

# Objects from ODB that are going to be placed, we also store the size and some tags that tell us what kind of relations
# that specific object allows
# tags = [on, in]
objects_of_interest = [ {"name": "op_microwave", "size": 0,  "tags": [True,     True]},
                        {"name": "red_mug",      "size": 0,  "tags": [False,    False]}]

object_of_interest_counter = 0

for obj in sample_surface_objects:
    # The loop starts with and UndoAfterExecution in order to clean up the cam poses from the previous iteration and
    # also remove the dropped objects and restore the sliced up objects.
    with bproc.utility.UndoAfterExecution():
        # Select the surfaces, where the object should be sampled on
        surface_obj = bproc.object.slice_faces_with_normals(obj)
        if surface_obj is None:
            continue

        surface_height_z = np.mean(surface_obj.get_bound_box(), axis=0)[2]
        
        
        

        dropped_object_list = []

        for name in objects_of_interest:

            # Take the object to be placed
            path_to_object = glob.glob(args.treed_obj_path + '/' + name["name"] + '/geometry/' + name["name"] + '*.obj')

            # Load the object, which should be sampled on the surface
            sampling_obj = bproc.loader.load_obj(path_to_object[0])

            tries = 0

            while tries < 1000:

                if sampling_obj[0].get_name() in bvh_cache:
                    del bvh_cache[sampling_obj[0].get_name()]

                # Sampling of the object
                dropped_object_list_temp = bproc.object.sample_poses_on_surface(sampling_obj, surface_obj, sample_pose,
                                                                                min_distance=0.1, max_distance=10,
                                                                                check_all_bb_corners_over_surface=True)
                # If the object couldn't be placed, jump to the next step
                if dropped_object_list_temp :

                    # Check if the sampling object has a collision with another object in the room
                    if CollisionUtility.check_intersections(sampling_obj[0], bvh_cache, room_objs, []):
                        break
                    else:
                        print("Collision detected, retrying!!!!!!!!!!!!!!!!!!")
                else:
                    break
                
                tries += 1

            # Save the objects dropped
            dropped_object_list.extend(dropped_object_list_temp)

        

        if not dropped_object_list:
            print("Dropping of the object failed")
            continue
        print("\n")
        print("################################################")
        for x in dropped_object_list:

            print(x.get_name())
        print("################################################")
        print("\n")

        # Enable physics for objects of interest (active) and the surface (passive)
        for dropped_object in dropped_object_list:
            dropped_object.enable_rigidbody(True)
        surface_obj.enable_rigidbody(False)

        # Run the physics simulation
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4,
                                                          check_object_interval=1)

        # join surface objects again
        surface_obj.join_with_other_objects([obj])

        # get the minimum value of all eight corners and from that the Z value
        min_coord_z = np.min(dropped_object.get_bound_box(local_coords=False), axis=0)[2]

        # Check if object is on surface, otherwise delete object
        objects_boxes = []
        remove_list = []
        for index, dropped_object in enumerate(dropped_object_list):
            
            print(f"Object: {dropped_object.get_name()} has a diff of: {abs(min_coord_z - surface_height_z)}m to the surface")
            # if distance is smaller than 5 cm delete for wrong positioning
            if abs(min_coord_z - surface_height_z) > 0.05:
                print("Delete this object, distance is above 0.05m")
                dropped_object.delete()
                remove_list.append(index)

            # remove deleted elements from dropped object list
            for ele in remove_list[::-1]:
                del dropped_object_list[ele]
                continue

            if not dropped_object_list:
                print(f"List is empty after removal of objects")
                # skip if no object is left
                continue

            # Set the custom propierty in the remaining objects of interest
            dropped_object.set_cp("category_id", object_of_interest_counter + 1)

            object_of_interest_counter += 1

            # Store the bounding boxes of the objects to calculate their size and location later
            objects_boxes.extend(dropped_object.get_bound_box())

        # place a camera

        # Point of interest to shoot
        poi = bproc.object.compute_poi(dropped_object_list)


        objects_location = np.mean(objects_boxes, axis=0)
        objects_size = np.max(np.max(objects_boxes, axis=0) - np.min(objects_boxes, axis=0))
        radius_min = objects_size * 1.5
        radius_max = objects_size * 10

        

        proximity_checks = {"min": radius_min, "avg": {"min": radius_min * 1.5, "max": radius_max * 0.8}, "no_background": True}
        cam_counter = 0
        
        # Init bvh tree containing all mesh objects
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(bproc.object.get_all_mesh_objects())
        for i in range(1000):
            camera_location = bproc.sampler.shell(center=objects_location, radius_min=radius_min, radius_max=radius_max,
                                                elevation_min=15, elevation_max=70)

            # Make sure that object is not always in the center of the camera
            toward_direction = (objects_location + np.random.uniform(0, 1, size=3) * objects_size * 0.5) - camera_location

            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - camera_location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)

            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree) \
                    and all(object in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15) for object in dropped_object_list):
                bproc.camera.add_camera_pose(cam2world_matrix)
                cam_counter += 1
            if cam_counter == 2:
                break
        if cam_counter == 0:
            raise Exception("No valid camera pose found!")

        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

        data = bproc.renderer.render()

        print(data)

        # write the data to a .hdf5 container
        

        bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                            instance_segmaps=data["instance_segmaps"],
                                            instance_attribute_maps=data["instance_attribute_maps"],
                                            colors=data["colors"],
                                            color_file_format="JPEG",
                                            )
        
        bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)
    break
