import blenderproc as bproc
import os
import numpy as np
import argparse

from typing import Dict
import mathutils
import sys
import json
# from matplotlib import pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument('treed_obj_path', help="Path to the downloaded 3D object")
parser.add_argument('output_dir', nargs='?', default="examples/automate_semantic_relations/Test_01/output", 
                    help="Path to where the final files, will be saved")
parser.add_argument('h5_file_name', default = "val.h5", help="Name of the file with the annotations")
parser.add_argument('--prioritize_relations', action="store_true", help="Place objects that allow relations")

parser.add_argument('--objects_focused', action="store_true", help="Take frames without deviation from the objects POI")
parser.add_argument('--include_base_object', action="store_true", help="Take the base objects as a relation generator")
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

def suitable_camera_poses(cycles: int, objects_location, objects_size: float, radius_min: float, radius_max: float,
                          visible_objects_threshold: float, dropped_object_list, cam_counter: int, objects_focused:bool):
    
    objects_on_all_frames = []

    proximity_checks = {"min": radius_min, "avg": {"min": radius_min , "max": radius_max }, "no_background": True}

    for i in range(cycles):
        # Place Camera
        camera_location = bproc.sampler.shell(center=objects_location, radius_min=radius_min, 
                                            radius_max=radius_max, elevation_min=0, elevation_max=15)

        if objects_focused:
            # Make sure that object is not always in the center of the camera
            toward_direction = (objects_location + np.random.uniform(0, 1, size=3) * objects_size * 0.4)
        else:
            toward_direction = objects_location
        # Compute rotation based on vector going from location towards poi/en/stable/strings.html


        rotation_matrix = bproc.camera.rotation_from_forward_vec(toward_direction - camera_location, 
                                                                inplane_rot=np.random.uniform(-0.349, 0.349))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)
        # print(np.sum([object in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15) 
        #                     for object in dropped_object_list])/len(dropped_object_list))

        objects_on_frame = []
        
        objects_on_frame = [object for object in dropped_object_list
                            if object in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15)]
        
        # if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree) \
        #     and visible_objects_threshold <= len(objects_on_frame) / len(dropped_object_list):
        if visible_objects_threshold <= len(objects_on_frame) / len(dropped_object_list):
            
            objects_on_all_frames.append(objects_on_frame)
            bproc.camera.add_camera_pose(cam2world_matrix)
            cam_counter += 1
            print(f"One camera pose looking at least to {visible_objects_threshold * 100} % of the interest objects has been stored")
        if cam_counter == 2:
            break

    return cam_counter, objects_on_all_frames


# Cache to fasten data Collection
bvh_cache : Dict[str, mathutils.bvhtree.BVHTree] = {}

# define the camera intrinsics
bproc.camera.set_resolution(640, 480)

desired_number_of_camera_poses = 2

# Select the objects, where other objects should be sampled on
sample_surface_objects = []
for room_obj in room_objs:
    
    if "table" in room_obj.get_name().lower() or "desk" in room_obj.get_name().lower(): 
    # if "dining table" in room_obj.get_name().lower(): 
    # if "table.003" == room_obj.get_name().lower(): 
        sample_surface_objects.append(room_obj)

    room_obj.set_cp("category_id", 0)


# Objects from ODB that are going to be placed, we also store the size and some tags that tell us what kind of relations
# that specific object allows
# tags = [on, in]

objects_of_interest = [ {"name": "op_microwave",        "tags": [True,     True],       "components": "door", 
                         "path": os.path.join(args.treed_obj_path,"op_microwave/geometry/op_microwave_open_glass.obj"),
                         "surface_distance": 0.02},
                        {"name": "red_mug",             "tags": [False,    False],      "components": False,
                         "path": os.path.join(args.treed_obj_path,"red_mug/geometry/red_mug.obj"),
                         "surface_distance": False },
                        {"name": "leather_tray",        "tags": [True,    False],       "components": False,
                         "path": os.path.join(args.treed_obj_path,"leather_tray/geometry/render.obj"),
                          "surface_distance": 0.005}]




placed_obj_counter = 0

for  base_obj in sample_surface_objects:

    
    
    # The loop starts with and UndoAfterExecution in order to clean up the cam poses from the previous iteration and
    # also remove the dropped objects and restore the sliced up objects.
    
    with bproc.utility.UndoAfterExecution():

        store_relations_and_features = {"relation": [], "attribute": []} # store_relations_and_features of the placed objects
                                                                        # attributes like "open" for the microwave

        # Select the surfaces, where the object should be sampled on
        objects_boxes = []
        dropped_object_list = []
        placed_obj_counter_static = placed_obj_counter
        
        base_object_dict = {"name": False, "tags": [True,    False], "surface_distance": False}
        
        
        placed_obj_counter = bproc.object.sample_scene_graph(base_obj, base_object_dict, objects_of_interest, objects_boxes, 
                                                                    dropped_object_list, placed_obj_counter, bvh_cache, room_objs,
                                                                    store_relations_and_features, verbose=False,max_n_tries=8, 
                                                                    max_n_obj=4, dropped_objects_types=[], 
                                                                    include_base_object=args.include_base_object,
                                                                    prioritize_relations=args.prioritize_relations)
        
        if not dropped_object_list:
            continue

        print("================================================")
        print(f"{len(dropped_object_list)} objects were placed")
        print("================================================")

        # Counter of the stored camera positions
        cam_counter = 0

        # Init bvh tree containing all mesh objects
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(bproc.object.get_all_mesh_objects())
        print(base_obj.get_name())
        print("Objects of interest: *********************************")
        for object_listed in dropped_object_list:
            print(object_listed.get_name())

        print("******************************************************")

        
        # Find a point of interest in the frame to focus, mean of all the bounding boxes of the objects
        objects_location = np.mean(objects_boxes, axis=0)
        objects_size = np.max(np.max(objects_boxes, axis=0) - np.min(objects_boxes, axis=0))
        print(f"objects_size: {objects_size}")
        if objects_size <= 0.50:
            radius_min = objects_size * 3 
            radius_max = objects_size * 4
        elif objects_size <= 0.80:
            radius_min = objects_size * 1.5
            radius_max = objects_size * 2
        else:
            radius_min = objects_size / 1.5
            radius_max = objects_size * 2
            
        if args.include_base_object:
            # Set the custom property in the the base object (the table or desk)
            base_obj.set_cp("category_id", placed_obj_counter + 1)
            placed_obj_counter += 1
            dropped_object_list.append(base_obj)
            # Store the type of relation of the dropped object
            store_relations_and_features["relation"].append("NONE")
            # If the object has a special characteristic that needs to be described (microwave open)
            store_relations_and_features["attribute"].append(float(0.0))

        objects_on_frames = []

        tries = [300, 200, 150, 50]
        thresholds = [1, 0.85, 0.75, 0.7]

        # tries = [300]
        # thresholds = [1]

        for number_of_cycles, visible_objects_threshold in zip(tries, thresholds):
            
            cam_counter, objects_on_frames_temp = suitable_camera_poses(number_of_cycles, objects_location, objects_size, radius_min, radius_max,
                                                                        visible_objects_threshold, dropped_object_list, cam_counter, args.objects_focused)
            objects_on_frames.extend(objects_on_frames_temp)
            if cam_counter == desired_number_of_camera_poses:
                break
        
        if cam_counter == 0:
            print(f"Image with the object {base_obj.get_name()} as a main parent has been skipped, since there are no suitable camera poses")
            continue
            # raise Exception("No valid camera pose found!")


        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

        data = bproc.renderer.render()

        

        
        bproc.writer.write_scene_graph(args.output_dir, args.h5_file_name, dropped_object_list, objects_on_frames, 
                                       data, store_relations_and_features)

        bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)
        


        


