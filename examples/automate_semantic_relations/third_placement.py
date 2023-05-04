import blenderproc as bproc
import os
import numpy as np
import argparse
from blenderproc.python.utility.CollisionUtility import CollisionUtility
from typing import Dict
import mathutils
from matplotlib import pyplot as plt
import h5py

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



def place_object(obj_to_place: bproc.types.MeshObject, receiving_obj: bproc.types.MeshObject, 
                 bvh_cache: Dict[str, mathutils.bvhtree.BVHTree], room_objs: list[bproc.types.MeshObject], 
                 dropped_objects: list[bproc.types.MeshObject], receiving_obj_attributes):

    """ Function to place an object on other, considering that the base or receiving object could have 
     prestablished objects on it. This function places the object and tests if it is close enough to
     the surface """
    

    was_putted_on = False
    was_putted_inside = False

    if receiving_obj_attributes["tags"][0] and not receiving_obj_attributes["tags"][1]: #np.random.randint(0, 2) == 1: #(len(receiving_obj_attributes["tags"])):
        surface_obj = bproc.object.slice_faces_with_normals(receiving_obj)
        was_putted_on = True

    elif receiving_obj_attributes["tags"][1]: #and np.random.randint(0, 2) == 1:
        with open('examples/automate_semantic_relations/heights.txt', 'w') as f:
            height = np.min(receiving_obj.get_bound_box()[:,2])
            f.write( f"[ {height}]")#, {height}, {height + 0.15}, {height + 0.2}, {height + 0.3}, {height + 0.5}
        f.close()
        
        surface_obj = bproc.object.extract_floor([receiving_obj], 
                                                 height_list_path="examples/automate_semantic_relations/heights.txt",
                                                 compare_height=0.03)[0]
        was_putted_inside = True
    else:
        return None, None
    if surface_obj is None:
        return None, None

    surface_height_z = np.max(surface_obj.get_bound_box(), axis=0)[2]

    def sample_pose(obj2: bproc.types.MeshObject):
        # Sample the spheres location above the surface
        obj2.set_location(bproc.sampler.upper_region(
            objects_to_sample_on=[surface_obj],
            min_height=0.3,
            max_height=0.4,
            use_ray_trace_check=False
        ))
        #Randomized rotation of the sampled object
        obj2.set_rotation_euler(bproc.sampler.uniformSO3(around_y=False, around_x=False, around_z=True))

    tries = 0

    objects_to_check = room_objs + dropped_objects

    for counter1, obj_to_check in enumerate(room_objs):
        print(counter1, obj_to_check.get_name())

    for counter2, obj_to_check in enumerate(dropped_objects):
        print(counter2, obj_to_check.get_name())

    while tries < 10:

        if obj_to_place[0].get_name() in bvh_cache:
            del bvh_cache[obj_to_place[0].get_name()]

        

        # Sampling of the object
        dropped_object_list_temp = bproc.object.sample_poses_on_surface(obj_to_place, surface_obj, sample_pose,
                                                                        min_distance=0.1, max_distance=10,
                                                                        check_all_bb_corners_over_surface=True,
                                                                        objects_to_check=objects_to_check )
        # If the object couldn't be placed, jump to the next step

        print("########## ¿Es aquí????? 1")

        if dropped_object_list_temp :

            # Check if the sampling object has a collision with another object in the room
            # if CollisionUtility.check_intersections(obj_to_place[0], bvh_cache, room_objs + dropped_objects, []):
            #     break
            # else:
            #     print("Collision detected, retrying!!!!!!!!!!!!!!!!!!")
            algo = 1
        else:
            # join surface objects again
            receiving_obj.join_with_other_objects([surface_obj])
            surface_obj.delete()
            print("Object couldn't be placed")
            return None, None
        
        tries += 1

        print("########## ¿Es aquí????? 2")
    # Enable physics for objects of interest (active) and the surface (passive)
    # If the object already has the rigid body features enabled, disable those and set the desired behavior
    
    # if dropped_object_list_temp[0].has_rigidbody_enabled():
    #     dropped_object_list_temp[0].disable_rigidbody()
    #     dropped_object_list_temp[0].enable_rigidbody(True)
    # else:
    #     dropped_object_list_temp[0].enable_rigidbody(True)

    # if receiving_obj.has_rigidbody_enabled():
    #     receiving_obj.disable_rigidbody()
    #     receiving_obj.enable_rigidbody(False, collision_shape='MESH')
    # else:
    #     receiving_obj.enable_rigidbody(False, collision_shape='MESH')

    # if surface_obj.has_rigidbody_enabled():
    #     surface_obj.disable_rigidbody()
    #     surface_obj.enable_rigidbody(False)
    # else:
    #     surface_obj.enable_rigidbody(False)



    # Run the physics simulation
    # bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4,
    #                                                     check_object_interval=1)

    # join surface objects again
    receiving_obj.join_with_other_objects([surface_obj])
    surface_obj.delete()

    # get the minimum value of all eight corners and from that the Z value
    min_coord_z = np.min(dropped_object_list_temp[0].get_bound_box(local_coords=False), axis=0)[2]


    # Check if object is on surface, otherwise delete object

    print("########## ¿Es aquí????? 3")
        
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
    
    # Store the generated relations between the two objects

    print("########## ¿Es aquí????? 4")
    if was_putted_on:
        relation = f"ON {receiving_obj.get_name()}"
    elif was_putted_inside:
        relation = f"INSIDE {receiving_obj.get_name()}"
    
    return relation, dropped_object_list_temp


def door_sampling(base: bproc.types.MeshObject, door: bproc.types.MeshObject, objects_boxes):

    door.set_location( base.get_origin() )

    for i in range(100):
        door.set_rotation_euler(base.get_rotation() + [0, 0, np.random.uniform(0, np.pi/2.6)]) # from 0 to pi/2.5
        if CollisionUtility.check_intersections(door, bvh_cache, room_objs, []):
            break
        else:
            door.set_rotation_euler(base.get_rotation() + [0, 0, np.pi/2])

    base.join_with_other_objects([door])
    
    

def adding_new_object(parent: bproc.types.MeshObject, parent_attributes: Dict, child_attributes: Dict, objects_boxes, 
                      dropped_object_list: list, category_counter, bvh_cache, room_objs, relations_and_attributes, abort_next_placement = False):
    """ 
    parent: 
    child_attributes:           Dictionary with the objects attributes
    parent_attributes:     Dictionary with the attributes of the surface 
        """
    
    for ii in range(10): #range(np.random.randint(1, 6)):
        # Load the object, which should be sampled on the surface
        child_attributes = objects_of_interest[np.random.randint(0, len(objects_of_interest))]
        sampling_obj = bproc.loader.load_obj(child_attributes["path"])

        if child_attributes["components"]:

            component = sampling_obj[1] # Maybe the name should be changed

            sampling_obj = [sampling_obj[0]]


        # Uniform probability calculated with the tags that are true, if several relations are posible, each one have the same probability
        # as the others
        if np.sum(parent_attributes["tags"]) >= 1:# and np.random.randint(1, np.sum(parent_attributes["tags"]) + 1) == 1:
            print(sampling_obj[0].get_name() + " _________________________________")
            relation, dropped_object = place_object(sampling_obj, parent, bvh_cache, room_objs, dropped_object_list, parent_attributes)

            if not dropped_object:
                print("222222222222222222222222222222222222222")
                abort_next_placement = True
                continue

        else:
            print("333333333333333333333333333333333333333")
            abort_next_placement = True
            return category_counter, abort_next_placement
        
        # Set the custom property in the remaining objects of interest
        dropped_object[0].set_cp("category_id", category_counter + 1)
        category_counter += 1
        dropped_object_list.extend(dropped_object)
        # Store the type of relation of the dropped object
        relations_and_attributes["relation"].append(relation)
        # If the object has a special characteristic that needs to be described (microwave open)
        relations_and_attributes["attribute"].append(child_attributes["components"] != False)

        # Store the bounding boxes of the objects to calculate their size and location later
        objects_boxes.extend(dropped_object[0].get_bound_box())

        if not abort_next_placement:

            next_child = objects_of_interest[np.random.randint(1, len(objects_of_interest))]

            category_counter, abort_next_placement = adding_new_object(sampling_obj[0], child_attributes, next_child, objects_boxes, 
                                                                            dropped_object_list, category_counter, bvh_cache, room_objs,
                                                                            relations_and_attributes) 
                 
            print("44444444444444444444444444444444444444")

        # if child_attributes["components"] == "door":

        #     door_sampling(sampling_obj[0], component, objects_boxes)
            
        # # Set the custom in the remaining objects of interest
        # dropped_object[0].set_cp("category_id", category_counter + 1)
        # category_counter += 1
        # dropped_object_list.extend(dropped_object)
        # # Store the bounding boxes of the objects to calculate their size and location later
        # objects_boxes.extend(dropped_object[0].get_bound_box())

        
    
    return  category_counter, abort_next_placement




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
                         "path": os.path.join(args.treed_obj_path,"op_microwave/geometry/op_microwave_open.obj")},
                        {"name": "red_mug",             "tags": [False,    False],      "components": False,
                         "path": os.path.join(args.treed_obj_path,"red_mug/geometry/red_mug.obj")}]

# objects_of_interest = [ {"name": "desk",                "tags": [True,    False],       "components": False},
#                         {"name": "table",               "tags": [True,    False],       "components": False},
#                         {"name": "op_microwave",        "tags": [True,     True],       "components": "door"},
#                         {"name": "red_mug",             "tags": [False,    False],      "components": False}]



# for ooi in objects_of_interest:

#             # Take the path to object to be placed
#             ooi["path"] = glob.glob(args.treed_obj_path + '/' + ooi["name"] + '/geometry/' + ooi["name"] + '*.obj')[-1]



store_relations_and_features = {"relation": [], "attribute": []}

object_of_interest_counter = 0

annotations = h5py.File(os.path.join(args.treed_obj_path,"val.h5"), 'w')

for scene_number, obj in enumerate(sample_surface_objects):
    # The loop starts with and UndoAfterExecution in order to clean up the cam poses from the previous iteration and
    # also remove the dropped objects and restore the sliced up objects.
    
    with bproc.utility.UndoAfterExecution():
        # Select the surfaces, where the object should be sampled on
        objects_boxes = []
        dropped_object_list = []

        # for ooi in objects_of_interest[:-1]:

        #     object_of_interest_counter, _ = adding_new_object(obj, {"tags": [True, False]}, ooi, objects_boxes, dropped_object_list, 
        #                                                         object_of_interest_counter, bvh_cache, room_objs)
        object_of_interest_counter, _ = adding_new_object(obj, {"tags": [True, False]}, objects_of_interest, objects_boxes, dropped_object_list, 
                                                                object_of_interest_counter, bvh_cache, room_objs, store_relations_and_features)
        

        if not dropped_object_list:
            continue
        

        # place a camera

        # Point of interest to shoot
        poi = bproc.object.compute_poi(dropped_object_list)

        objects_location = np.mean(objects_boxes, axis=0)

        objects_size = np.max(np.max(objects_boxes, axis=0) - np.min(objects_boxes, axis=0))
        radius_min = objects_size / 2
        radius_max = objects_size * 10

        

        proximity_checks = {"min": radius_min, "avg": {"min": radius_min * 2 , "max": radius_max }, "no_background": True}
        cam_counter = 0
        print("================================================")
        print(len(dropped_object_list))
        # Init bvh tree containing all mesh objects
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(bproc.object.get_all_mesh_objects())
        for i in range(1000):
            camera_location = bproc.sampler.shell(center=objects_location, radius_min=radius_min, radius_max=radius_max,
                                                elevation_min=0, elevation_max=75)

            # Make sure that object is not always in the center of the camera
            toward_direction = (objects_location + np.random.uniform(0, 1, size=3) * objects_size * 0.5) - camera_location

            # Compute rotation based on vector going from location towards poi/en/stable/strings.html


            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - camera_location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)

            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree) \
                    and 0.5 <= np.sum([object in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15) 
                                       for object in dropped_object_list])/len(dropped_object_list):
            # if 0.2 <= np.sum([object in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=15) for object in dropped_object_list])/len(dropped_object_list):
                bproc.camera.add_camera_pose(cam2world_matrix)
                cam_counter += 1
            if cam_counter == 2:
                break
        if cam_counter == 0:
            raise Exception("No valid camera pose found!")

        
        # Set the custom property in the the base object (the table or desk)
        obj.set_cp("category_id", object_of_interest_counter + 1)
        object_of_interest_counter += 1
        dropped_object_list.append(obj)
        # Store the type of relation of the dropped object
        store_relations_and_features["relation"].append("NONE")
        # If the object has a special characteristic that needs to be described (microwave open)
        store_relations_and_features["attribute"].append(False)



        data = bproc.renderer.render()
        
        # write the data to a .hdf5 container
        

        # bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
        #                                     instance_segmaps=data["instance_segmaps"],
        #                                     instance_attribute_maps=data["instance_attribute_maps"],
        #                                     colors=data["colors"],
        #                                     color_file_format="JPEG",
        #                                     )
        
        bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=True)



        


        objects_number = len(dropped_object_list)
        relations_number = 2

        attributes = []
        relations = np.array(relations_number * [-1 * np.eye(objects_number)])

        objects = [scene_object.get_name().encode('utf-8') for scene_object in dropped_object_list]

        print(objects)

        for counter, scene_object in enumerate(dropped_object_list):

            current_relation = store_relations_and_features["relation"][counter].split()

            if current_relation[0] != "NONE":
                parent_index = objects.index(" ".join(current_relation[1:]).encode('utf-8'))
                child_index = objects.index(scene_object.get_name().encode('utf-8'))

            elif current_relation[0] == "ON":
                
                relations[0, child_index, parent_index] = 1
            elif current_relation[0] == "INSIDE":
                
                relations[1, child_index, parent_index] = 1


        annotations.create_group(str(scene_number))
        annotations[str(scene_number)].create_dataset('attributes', data=store_relations_and_features["attribute"])
        annotations[str(scene_number)].create_dataset('bboxes', (objects_number, 4))

        annotations[str(scene_number)].create_dataset('image', data=np.array(data['colors'])) 

        annotations[str(scene_number)].create_dataset('objects', (objects_number,), data=np.array(objects))

        annotations[str(scene_number)].create_dataset('relations', data=np.array(relations))

        # plt.imshow(data['colors'][1], interpolation='nearest')
        # plt.show()
        # plt.savefig("mygraph.png")

        
    # break


