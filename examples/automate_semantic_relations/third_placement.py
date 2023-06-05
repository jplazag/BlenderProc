import blenderproc as bproc
import os
import bpy
import numpy as np
import argparse
from blenderproc.python.utility.CollisionUtility import CollisionUtility
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



def place_object(obj_to_place: bproc.types.MeshObject, surface_obj: bproc.types.MeshObject, 
                 room_objs: list[bproc.types.MeshObject], 
                 dropped_objects: list[bproc.types.MeshObject]):

    """ Function to place an object on other, considering that the base or receiving object could have 
     preestablished objects on it. This function places the object and tests if it is close enough to
     the surface """

    surface_height_z = np.max(surface_obj.get_bound_box(), axis=0)[2]

    def sample_pose(obj2: bproc.types.MeshObject):
        # Sample the objects location above the surface
        obj2.set_location(bproc.sampler.upper_region(
            objects_to_sample_on=[surface_obj],
            min_height=0.01,
            max_height=1,
            use_ray_trace_check=False,
            face_sample_range=[0.1, 0.9]
        ))
        #Randomized rotation of the sampled object
        obj2.set_rotation_euler(bproc.sampler.uniformSO3(around_y=False, around_x=False, around_z=True))

    tries = 0

    objects_to_check = room_objs + dropped_objects
    
    # Sampling of the object
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")


    dropped_object_list_temp = bproc.object.sample_poses_on_surface(obj_to_place, surface_obj, sample_pose,
                                                                    max_tries = 100, min_distance=0.1, max_distance=10,
                                                                    check_all_bb_corners_over_surface=False,
                                                                    objects_to_check=objects_to_check )

    sys.stdout = old_stdout # reset old stdout



    if not dropped_object_list_temp :
        
        print("Object couldn't be placed")
        return None, None
    

    # ! KeyError: 'bpy_prop_collection[key]: key "Floor" not found'

    # Enable physics for objects of interest (active) and the surface (passive)
    # If the object already has the rigid body features enabled, disable those and set the desired behavior
    
    
    dropped_object_list_temp[0].enable_rigidbody(True)

    surface_obj.enable_rigidbody(False)


    # Run the physics simulation
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4,
                                                        check_object_interval=1)


    # get the minimum value of all eight corners and from that the Z value
    min_coord_z = np.min(dropped_object_list_temp[0].get_bound_box(local_coords=False), axis=0)[2]


    # Check if object is on surface, otherwise delete object

        
    print(f"Object: {dropped_object_list_temp[0].get_name()} has a diff of: {abs(min_coord_z - surface_height_z)}m to the surface")
    # if distance is smaller than 5 cm delete for wrong positioning
    if abs(min_coord_z - surface_height_z) > 0.05:
        print("Delete this object, distance is above 0.05m")
        dropped_object_list_temp[0].delete()
        del dropped_object_list_temp[0]

    if not dropped_object_list_temp:
        print(f"The object couldn't be placed")
        # skip if no object is left
        return None, None
    
    
    dropped_object_list_temp[0].disable_rigidbody()
    surface_obj.disable_rigidbody()
    
    return dropped_object_list_temp


def door_sampling(base: bproc.types.MeshObject, door: bproc.types.MeshObject, room_objs: list[bproc.types.MeshObject], placed_objects: list[bproc.types.MeshObject]):

    door.set_location( base.get_origin() )

    for i in range(10):
        door
        rotation = np.random.uniform(0, np.pi/2.6)
        door.set_rotation_euler(base.get_rotation() + [0, 0, rotation]) # from 0 to pi/2.5
        if CollisionUtility.check_intersections(door, bvh_cache, room_objs + placed_objects, []):
            base.join_with_other_objects([door])
            return rotation

    rotation = np.pi/2
    door.set_rotation_euler(base.get_rotation() + [0, 0, rotation])

    base.join_with_other_objects([door])
    return rotation
    

    

    


def calculate_area_of_surface(object_to_measure: bproc.types.MeshObject, x_vector=np.array([1,0,0]), y_vector=np.array([0,1,0])):

    box = object_to_measure.get_bound_box()

    x_length = max(x_vector.dot(corner) for corner in box) - min(x_vector.dot(corner) for corner in box)
    y_length = max(y_vector.dot(corner) for corner in box) - min(y_vector.dot(corner) for corner in box)

    return x_length * y_length


def adding_new_object(parent: bproc.types.MeshObject, parent_attributes: Dict, children_attributes: Dict, objects_boxes, 
                      dropped_object_list: list, category_counter, bvh_cache, room_objs, relations_and_attributes,
                      objects_to_look_for):
    """ Recursive function that insert objects (one after another if the base object so allow it) in different configurations
    (ON, INSIDE) 

    parent:                         Base object where the new object is going to be placed
    children_attributes:            Dictionary with the objects attributes
    parent_attributes:              Dictionary with the attributes of the surface 
    objects_boxes:                  Box of each object, which cover all the volume of the object
    category_counter:               Counter to set the custom property of each object
    bvh_cache:                      BVH tree to store the placed objects and reduce computation time
    room_objs:                      Default objects in the room
    relations_and_attributes:       Dictionary that allocates the relations that each object has and some attributes

    return category_counter         Counter of the custom property of all the placed objects
        """
    
    was_putted_on = False
    was_putted_inside = False

    test = False # Variable to switch to a test environment in which one could reduce the amount of objects
                 # and generate specific configuration to analyse a certain behavior
    parent_is_a_desk = False

    # Probability  of starting the placement process when the parent object can host at least one possible relation
    if True in parent_attributes["tags"] : 
            
        if np.random.uniform(0,1) <= 0.7: # 70% of probability to add a new object

            if test == True:
                # Array with al the types of relations that the parent object could host

                if parent_attributes["tags"][0] and not parent_attributes["tags"][1]:
                    parent_is_a_desk = True
                    relations_to_establish = [0]
                else:
                    relations_to_establish = [1]
                

            else:
                # Array with al the types of relations that the parent object could host
                possible_relations =  [element for element in range(len(parent_attributes["tags"])) 
                                    if parent_attributes["tags"][element] != False]
                # Select randomly one of those relations
                relations_to_establish = np.random.choice(possible_relations, 
                                                          np.random.randint(1, len(possible_relations) + 1), 
                                                          replace=False) # 0 = ON; 1 = INSIDE

            for relation_to_establish in relations_to_establish:
            
                if relation_to_establish == 0: # If == 0 take the upper surface of the parent; ON relation
                    print(f"The surface is going to be taken from {parent.get_name()}")
                    surface_obj = bproc.object.slice_faces_with_normals(parent) 
                    was_putted_on = True

                elif relation_to_establish == 1: # If == 1 take the inner surface of the parent; INSIDE relation
                    height = np.min(parent.get_bound_box()[:,2]) + parent_attributes["surface_distance"]
                    # with open('examples/automate_semantic_relations/heights.txt', 'w') as f:
                    #     height = np.min(parent.get_bound_box()[:,2]) + parent_attributes["surface_distance"]
                    #     f.write( f"[ {height}]")
                    # f.close()
                    # Variable to store all the objects placed inside, so we could check if they are visible
                    objects_to_look_for[parent.get_name()] = [parent] # Start with the parent to use it later
                                                                      # in the camera pose generation
                    print(f"The surface is going to be taken from {parent.get_name()}")
                    # surface_obj = bproc.object.extract_floor([parent], 
                    #                                         height_list_path="examples/automate_semantic_relations/heights.txt",
                    #                                         compare_height=0.01, new_name_for_object="Surface")[0]
                    surface_obj = bproc.object.extract_floor([parent], 
                                                            height_list=[height],
                                                            compare_height=0.01, new_name_for_object="Surface")[0]
                    was_putted_inside = True

                if surface_obj is None: # If a surface couldn't be found
                    return category_counter
            
                # Try to place 10 objects in or on the actual parent

                if parent_is_a_desk:
                    max_n_obj = 2
                    n = 0
                else:
                    max_n_obj = 8
                    n = 1

                for ii in range(np.random.randint(1, max_n_obj)):

                    # Load the object, which should be sampled on the surface
                    child_attributes = children_attributes[np.random.randint(0, len(children_attributes) )] #n]
                    sampling_obj = bproc.loader.load_obj(child_attributes["path"])
                    actual_attribute = 0

                    print("--------------------------------------------------------------")
                    
                    print(f"The actual parent object is {parent.get_name()}")
                    print(f"And the relation(s) that are going to be generate is/are {relation_to_establish}")
                    print(f"This is the child number {ii}")
                    print(f"The name of that child object is {sampling_obj[0].get_name()}")
                    
                    if child_attributes["components"]:

                        component = sampling_obj[1] # Maybe the name should be changed
                        sampling_obj = [sampling_obj[0]]
                    # Compare the size of the surface with the base of the bounding box from the sampling_obj
                    surface_area = calculate_area_of_surface(surface_obj)
                    sampling_obj_area = calculate_area_of_surface(sampling_obj[0], y_vector=np.array([0,0,1]))

                    print(f"The base of the child object is {round(sampling_obj_area, 3)} m^2")
                    print(f"The area of the parent surface is {round(surface_area, 3)} m^2")

                    if surface_area <= sampling_obj_area + 0.06:
                        sampling_obj[0].delete()
                        print("--------------------------------------------------------------")
                        continue

                    dropped_object = place_object(sampling_obj, surface_obj, room_objs, dropped_object_list)

                        

                    if not dropped_object[0]: # If the sampling_obj couldn't be placed
                        print("sampling_obj couldn't be placed")
                        print("--------------------------------------------------------------")
                        continue

                    print("--------------------------------------------------------------")

                    # Store the type of relation of the dropped object
                    if was_putted_on:
                        relation = f"ON {parent.get_name()}"
                    elif was_putted_inside:
                        parent_name = parent.get_name()
                        relation = f"INSIDE {parent_name}"
                        objects_to_look_for[parent_name].append(dropped_object[0])

                    

                    # Store the bounding boxes of the objects to calculate their size and location later
                    objects_boxes.extend(dropped_object[0].get_bound_box())

                    # Prepare the next child to place
                    # next_child = children_attributes[np.random.randint(0, len(children_attributes))]


                    category_counter = adding_new_object(sampling_obj[0], child_attributes, children_attributes, objects_boxes, dropped_object_list, 
                                                        category_counter, bvh_cache, room_objs, relations_and_attributes,
                                                        objects_to_look_for) 
                    
                    if child_attributes["components"] == "door":

                        door_rotation = door_sampling(sampling_obj[0], component, room_objs, dropped_object_list )
                        if door_rotation < np.pi/4:
                            actual_attribute = 1.0 # The door is open

                    # Set the custom property in the remaining objects of interest
                    dropped_object[0].set_cp("category_id", category_counter + 1)
                    category_counter += 1
                    dropped_object_list.extend(dropped_object)

                    relations_and_attributes["relation"].append(relation)
                    # If the object has a special characteristic that needs to be described (microwave open)
                    relations_and_attributes["attribute"].append(actual_attribute)
                was_putted_inside = False
                was_putted_on = False
                # join surface objects again
                parent.join_with_other_objects([surface_obj])

                # ! Blender crash
                # surface_obj.delete()
                # Join the component 
                
    
    return  category_counter





def visible_objects_considering_glass(cam2world_matrix, sqrt_number_of_rays: int = 10):
    """ Returns a set of objects visible from the given camera pose, including the objects that are behind a 
    transparent screen, like a window.

    Sends a grid of rays through the camera frame and returns all objects hit by at least one ray.

    :param cam2world_matrix: The world matrix which describes the camera orientation to check.
    :param sqrt_number_of_rays: The square root of the number of rays which will be used to determine the
                                visible objects.
    :return: A set of objects visible hit by the sent rays.
    """

    

    cam2world_matrix = mathutils.Matrix(cam2world_matrix)

    visible_objects_set = set()
    cam_ob = bpy.context.scene.camera
    cam = cam_ob.data
    # Focal length to convert form m to normalized coordinates
    focal_length = cam_ob.data.lens

    # Get position of the corners of the near plane
    frame = cam.view_frame(scene=bpy.context.scene)
    # Bring to world space
    frame = [cam2world_matrix @ v for v in frame]

    # Compute vectors along both sides of the plane
    vec_x = frame[1] - frame[0]
    vec_y = frame[3] - frame[0]
    # Go in discrete grid-like steps over plane
    position = cam2world_matrix.to_translation()
    # Store the glass_like materials hit
    materials_hit = []
    glass_like_material_name = ""
    for x in range(0, sqrt_number_of_rays):
        for y in range(0, sqrt_number_of_rays):
            step = 0
            we_are_on_glass_r = True
            we_are_on_glass_l = True
            # Compute current point on plane
            end = frame[0] + vec_x * x / float(sqrt_number_of_rays - 1) + vec_y * y / float(sqrt_number_of_rays - 1)

            _, _, _, face_index, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.evaluated_depsgraph_get(),
                                                                    position, end - position)
            if hit_object:
                visible_objects_set.add(MeshObject(hit_object))
            
            # if hit_object:
            #     material_index = hit_object.data.polygons[face_index].material_index
            #     hit_material_name =  hit_object.material_slots.keys()[material_index]

            #     if "MTL1" in hit_material_name and hit_material_name not in materials_hit:
            #         print(hit_material_name)

            while we_are_on_glass_r and we_are_on_glass_l:

                # Send ray from the camera position through the current point on the plane

                for direction in range(2):
                    if (we_are_on_glass_r and direction == 0) or (we_are_on_glass_l and direction == 1):
                        end_search = end + (-1)**direction * vec_x * (1/ (2 * focal_length) ) * step + (-1)**direction * vec_y * (1/ (2 *focal_length) ) * step
                        _, hit_location, _, face_index, hit_object_search, _ = bpy.context.scene.ray_cast(bpy.context.evaluated_depsgraph_get(),
                                                                            position, end_search - position)
                        # Add hit object to set
                        if hit_object_search:


                            
                            new_hit_location = hit_location + (end_search - position) / np.linalg.norm(end_search - position) * 0.06
                            # hit_object = MeshObject(hit_object)
                            material_index = hit_object_search.data.polygons[face_index].material_index
                            
                            if len(hit_object_search.material_slots.keys()) < 1:
                                if direction == 0:
                                    we_are_on_glass_r = False
                                else:
                                    we_are_on_glass_l = False
                                continue

                            hit_material_name =  hit_object_search.material_slots.keys()[material_index]

                            if "MTL1" in hit_material_name and hit_material_name not in materials_hit: #! Generalize the name of the material
                                glass_like_material_name = hit_material_name
                                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                                _, hit_location, _, face_index, hit_object2, _ = bpy.context.scene.ray_cast(bpy.context.evaluated_depsgraph_get(),
                                                                                        new_hit_location, end_search - position)
                                if hit_object2:
                                    visible_objects_set.add(MeshObject(hit_object2))
                                    # material_index = hit_object2.data.polygons[face_index].material_index
                            
                                    # if len(hit_object2.material_slots.keys()) < 1:
                                    #     continue

                                    # hit_material_name =  hit_object2.material_slots.keys()[material_index]
                                    # print("###########################3")
                                    # print(hit_material_name)
                                
                            else:
                                if direction == 0:
                                    we_are_on_glass_r = False
                                else:
                                    we_are_on_glass_l = False

            

                step += 1

            

            materials_hit.append(glass_like_material_name)
            

    return visible_objects_set



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
        
        object_of_interest_counter = adding_new_object(obj, {"tags": [True, False]}, objects_of_interest, objects_boxes, dropped_object_list, 
                                                        object_of_interest_counter, bvh_cache, room_objs, store_relations_and_features,
                                                        objects_to_search_inside)
        

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


        


