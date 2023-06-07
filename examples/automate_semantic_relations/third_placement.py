import blenderproc as bproc
import os
import numpy as np
import argparse

from typing import Dict
import mathutils
import sys
# import json
# from matplotlib import pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument('treed_obj_path', help="Path to the downloaded 3D object")
parser.add_argument('output_dir', nargs='?', default="examples/automate_semantic_relations/Test_01/output", 
                    help="Path to where the final files, will be saved")
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
                          visible_objects_threshold: float, dropped_object_list, cam_counter: int):
    for i in range(cycles):
        # Place Camera
        camera_location = bproc.sampler.shell(center=objects_location, radius_min=radius_min, 
                                            radius_max=radius_max, elevation_min=0, elevation_max=15)

        # Make sure that object is not always in the center of the camera
        toward_direction = (objects_location + np.random.uniform(0, 1, size=3) * objects_size * 0.5)

        # Compute rotation based on vector going from location towards poi/en/stable/strings.html


        rotation_matrix = bproc.camera.rotation_from_forward_vec(toward_direction - camera_location, 
                                                                inplane_rot=np.random.uniform(-0.349, 0.349))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)
        # print(np.sum([object in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15) 
        #                     for object in dropped_object_list])/len(dropped_object_list))
        

        if visible_objects_threshold <= np.sum([object in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15) 
                                for object in dropped_object_list])/len(dropped_object_list):

            bproc.camera.add_camera_pose(cam2world_matrix)
            cam_counter += 1
            print(f"One camera pose looking at least to {visible_objects_threshold * 100} % of the interest objects has been stored")
        if cam_counter == 2:
            break

    return cam_counter


# Cache to fasten data Collection
bvh_cache : Dict[str, mathutils.bvhtree.BVHTree] = {}

# define the camera intrinsics
bproc.camera.set_resolution(512, 512)

# Select the objects, where other objects should be sampled on
sample_surface_objects = []
for room_obj in room_objs:
    
    # if "table" in room_obj.get_name().lower() or "desk" in room_obj.get_name().lower(): 
    if "dining table" in room_obj.get_name().lower(): 
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
                        {"name": "leather_tray",        "tags": [False,    True],       "components": False,
                         "path": os.path.join(args.treed_obj_path,"leather_tray/geometry/render.obj"),
                          "surface_distance": 0.005}]


store_relations_and_features = {"relation": [], "attribute": []} # store_relations_and_features of the placed objects
                                                                 # attributes like "open" for the microwave

placed_obj_counter = 0

for  base_obj in sample_surface_objects:

    
    # The loop starts with and UndoAfterExecution in order to clean up the cam poses from the previous iteration and
    # also remove the dropped objects and restore the sliced up objects.
    
    with bproc.utility.UndoAfterExecution():
        # Select the surfaces, where the object should be sampled on
        objects_boxes = []
        dropped_object_list = []
        
        placed_obj_counter = bproc.object.sample_scene_graph(base_obj, {"tags": [True, False]}, objects_of_interest, objects_boxes, 
                                                                     dropped_object_list, placed_obj_counter, bvh_cache, 
                                                                     room_objs, store_relations_and_features, verbose=False)
        
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

        if cam_counter < 6:
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
            # proximity_checks = {"min": radius_min, "avg": {"min": radius_min , "max": radius_max * 2 }, "no_background": True}

            for number_of_cycles, visible_objects_threshold in zip([300, 200, 150, 50], [1, 0.85, 0.75, 0.7]):
                
                cam_counter = suitable_camera_poses(number_of_cycles, objects_location, objects_size, radius_min, radius_max,
                                                    visible_objects_threshold, dropped_object_list, cam_counter)
                if cam_counter == 2:
                    break
            
            if cam_counter == 0:
                print(f"Image with the object {base_obj.get_name()} as a main parent has been skipped, since there are no suitable camera poses")
                continue
                # raise Exception("No valid camera pose found!")

        
        # Set the custom property in the the base object (the table or desk)
        base_obj.set_cp("category_id", placed_obj_counter + 1)
        placed_obj_counter += 1
        dropped_object_list.append(base_obj)
        # Store the type of relation of the dropped object
        store_relations_and_features["relation"].append("NONE")
        # If the object has a special characteristic that needs to be described (microwave open)
        store_relations_and_features["attribute"].append(float(0.0))

        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

        data = bproc.renderer.render()

        h5_file_name = "val.h5"
        bproc.writer.write_scene_graph(args.output_dir, h5_file_name, dropped_object_list, data, 
                                    store_relations_and_features, cam_counter, test_bboxes=True)

        bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)
        


        


