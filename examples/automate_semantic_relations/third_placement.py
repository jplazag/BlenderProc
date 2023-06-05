import blenderproc as bproc
import os
import bpy
import numpy as np
import argparse

from typing import Dict
import mathutils
from matplotlib import pyplot as plt
import sys
from blenderproc.python.types.MeshObjectUtility import MeshObject
import json

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

# define the camera intrinsics
bproc.camera.set_resolution(512, 512)

# Select the objects, where other objects should be sampled on
sample_surface_objects = []
for obj in room_objs:
    if "dining table" in obj.get_name().lower(): #if "table" in obj.get_name().lower() or "desk" in obj.get_name().lower():
        sample_surface_objects.append(obj)

    obj.set_cp("category_id", 0)


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


store_relations_and_features = {"relation": [], "attribute": []}

object_of_interest_counter = 0

""" annotations = h5py.File(os.path.join(args.output_dir,"val.h5"), 'a',track_order=True) """

objects_to_search_inside = {}


""" if list(annotations.keys()):

    next_scene = int(list(annotations["/"].keys())[-1]) + 1 
else:
    next_scene = 0 """

for  obj in sample_surface_objects:

    
    # The loop starts with and UndoAfterExecution in order to clean up the cam poses from the previous iteration and
    # also remove the dropped objects and restore the sliced up objects.
    
    with bproc.utility.UndoAfterExecution():
        # Select the surfaces, where the object should be sampled on
        objects_boxes = []
        dropped_object_list = []
        
        object_of_interest_counter = bproc.object.sample_scene_graph(obj, {"tags": [True, False]}, objects_of_interest, objects_boxes, 
                                                                     dropped_object_list, object_of_interest_counter, bvh_cache, 
                                                                     room_objs, store_relations_and_features, objects_to_search_inside)
        

        if not dropped_object_list:
            continue


        # Set the custom property in the the base object (the table or desk)
        obj.set_cp("category_id", object_of_interest_counter + 1)
        object_of_interest_counter += 1
        dropped_object_list.append(obj)
        # Store the type of relation of the dropped object
        store_relations_and_features["relation"].append("NONE")
        # If the object has a special characteristic that needs to be described (microwave open)
        store_relations_and_features["attribute"].append(float(0.0))

        print("================================================")
        print(f"{len(dropped_object_list)} objects were placed")
        print("================================================")

        # Counter of the stored camera positions
        cam_counter = 0

        # Init bvh tree containing all mesh objects
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(bproc.object.get_all_mesh_objects())

        for obj in dropped_object_list:
            print(obj.get_name())

        print("******************************************************")
        inside_objects = [] # Array to store all the objects placed inside without distinction
        if len(objects_to_search_inside.values()) > 0:
            for _, inside_group in objects_to_search_inside.items():
                
                inside_objects.extend(inside_group[1:])

                object_to_focus = inside_group[0] # Take the parent to use it as point of interest
                 # Point of interest to shoot
                poi = bproc.object.compute_poi([object_to_focus])
                
                object_size = np.max(np.max(object_to_focus.get_bound_box(), axis=0) - np.min(object_to_focus.get_bound_box(), axis=0))
                r_min = object_size
                r_max = object_size * 3
                proximity_checks = {"min": r_min, "avg": {"min": r_min , "max": r_max * 2 }, "no_background": True}
                
                for i in range(50):
                    #Place camera
                    camera_location = bproc.sampler.shell(center=poi, radius_min=r_min, radius_max=r_max,
                                                        elevation_min=0, elevation_max=20)

                    # Make sure that object is not always in the center of the camera
                    toward_direction = (poi + np.random.uniform(0, 1, size=3) * object_size * 0.5) - camera_location

                    # Compute rotation based on vector going from location towards poi/en/stable/strings.html
                    rotation_matrix = bproc.camera.rotation_from_forward_vec(toward_direction - camera_location, inplane_rot=np.random.uniform(-0.349, 0.349))
                    # Add homog cam pose based on location an rotation
                    cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)
                    
                    objects_seen = bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15)
                    
                    if len(inside_group) > 1:
                        # print(len(set(objects_seen) & set(inside_group[1:])) / len(inside_group[1:]))
                        if len(set(objects_seen) & set(inside_group[1:])) / len(inside_group[1:]) > 0.2:
                            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
                                bproc.camera.add_camera_pose(cam2world_matrix)
                                cam_counter += 1
                                print("One camera pose looking inside was stored")
                    if cam_counter > 3:
                        break

        first_cycle_cam_counter =  cam_counter
        if cam_counter < 6:
            

            # Find a point of interest in the frame to focus, mean of all the bounding boxes of the objects
            objects_location = np.mean(objects_boxes, axis=0)
            objects_size = np.max(np.max(objects_boxes, axis=0) - np.min(objects_boxes, axis=0))
            radius_min = objects_size * 2
            radius_max = objects_size * 3
            proximity_checks = {"min": radius_min, "avg": {"min": radius_min , "max": radius_max * 2 }, "no_background": True}
            
            for i in range(500):
                # Place Camera
                camera_location = bproc.sampler.shell(center=objects_location, radius_min=radius_min, 
                                                    radius_max=radius_max, elevation_min=0, elevation_max=15)

                # Make sure that object is not always in the center of the camera
                toward_direction = (objects_location + np.random.uniform(0, 1, size=3) * objects_size * 0.5) - camera_location

                # Compute rotation based on vector going from location towards poi/en/stable/strings.html


                rotation_matrix = bproc.camera.rotation_from_forward_vec(toward_direction - camera_location, 
                                                                        inplane_rot=np.random.uniform(-0.349, 0.349))
                # Add homog cam pose based on location an rotation
                cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)
                # print(np.sum([object in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15) 
                #                     for object in dropped_object_list])/len(dropped_object_list))

                objects_to_check = list(set(dropped_object_list) - set(inside_objects))
                

                if 0.6 <= np.sum([object in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15) 
                                    for object in objects_to_check])/len(objects_to_check):

                    bproc.camera.add_camera_pose(cam2world_matrix)
                    cam_counter += 1
                    print("One camera pose looking all the objects in general was stored")
                if cam_counter >= first_cycle_cam_counter + 2:
                    break
            if cam_counter == 0:
                raise Exception("No valid camera pose found!")

        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

        data = bproc.renderer.render()


        """ def numpy_encoder(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Numpy-Array in Python-Liste umwandeln
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        
        with open("daten.json", "w") as file:
            json.dump(data, file, default=numpy_encoder) """


        # plt.imshow(np.array(data["instance_segmaps"][0]), cmap='gray')
        # plt.axis('off')  # Die Achsenbeschriftungen ausblenden
        # plt.show()
        # plt.savefig("bild.png")  # Das Bild in eine Datei speichern

        h5_file_name = "val.h5"
        bproc.writer.write_scene_graph(os.path.join(args.output_dir,h5_file_name), dropped_object_list, data, store_relations_and_features, 
                                       cam_counter)


        


