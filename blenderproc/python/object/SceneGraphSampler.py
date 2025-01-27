
from typing import Dict, Optional
import mathutils
import numpy as np

from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.object.FaceSlicer import extract_floor, slice_faces_with_normals
from blenderproc.python.object.PhysicsSimulation import simulate_physics_and_fix_final_poses
from blenderproc.python.object.OnSurfaceSampler import sample_poses_on_surface
from blenderproc.python.loader.ObjectLoader import load_obj
from blenderproc.python.sampler.UpperRegionSampler import upper_region
from blenderproc.python.sampler.UniformSO3 import uniformSO3
from blenderproc.python.utility.CollisionUtility import CollisionUtility

def sample_scene_graph(parent: MeshObject, parent_attributes: Dict, children_attributes: Dict, objects_boxes: list, 
                      dropped_object_list: list, category_counter: int, bvh_cache: Optional[Dict[str, mathutils.bvhtree.BVHTree]],
                      room_objs: list[MeshObject], relations_and_attributes: Dict, verbose = True, max_n_tries: int = 8, max_n_obj: int = 15,
                      dropped_objects_types: list = [], include_base_object = False, prioritize_relations = False):
    """ Recursive function that insert objects (one after another if the base object so allow it) in different configurations
    (ON, INSIDE) 

    :param parent: Base object where the new object is going to be placed.
    :param children_attributes: Dictionary with the objects attributes.
    :param parent_attributes: Dictionary with the attributes of the surface.
    :param objects_boxes: Box of each object, which cover all the volume of the object.
    :param category_counter: Counter to set the custom property of each object.
    :param bvh_cache: BVH tree to store the placed objects and reduce computation time.
    :param room_objs: Default objects in the room.
    :param relations_and_attributes: Dictionary that allocates the relations that each object has and some attributes.
    :param max_n_tries: Number of objects, that are going to be tried to place pro parent 
    :param max_n_obj: Maximum number of objects in an scene.
    :param dropped_objects_types: Store the index of the placed object to know if a object type has already an instance
    :param include_base_object: Consider the base objects (tables, desks,...) as relation generators 
    :param prioritize_relations: Force the placement of objects, that could generate relations (microwaves, trays, ...)

    :return: Type: :class:`int`, category_counter, the counter of the custom property of all the placed objects.
        """
    
    was_putted_on = False
    was_putted_inside = False
    

    # Probability  of starting the placement process when the parent object can host at least one possible relation
    if True in parent_attributes["tags"] : 
            
        if np.random.uniform(0,1) >= 0.7: # 70% of probability to add a new object
            return category_counter
        # Array with al the types of relations that the parent object could host
        possible_relations =  [element for element in range(len(parent_attributes["tags"])) 
                            if parent_attributes["tags"][element]]
        # Select randomly one of those relations
        relations_to_establish = np.random.choice(possible_relations, 
                                                    np.random.randint(1, len(possible_relations) + 1), 
                                                    replace=False) # 0 = ON; 1 = INSIDE
        
        for relation_to_establish in relations_to_establish:

            if len(objects_boxes) / 8 >= max_n_obj:
                if verbose:
                        print("The maximal amount of objects was reached")
                break
        
            if relation_to_establish == 0: # If == 0 take the upper surface of the parent; ON relation
                # print(parent_attributes["surface_distance"])
                if parent_attributes["surface_distance"] and not parent_attributes["tags"][1]: #If has a distance for a surface and the INSIDE tag is false
                                                                                                # it means that the distance is for the ON relation
                    print(f"The surface is going to be taken from {parent.get_name()}")
                    height = np.min(parent.get_bound_box()[:,2]) + parent_attributes["surface_distance"]
                    surface_obj = extract_floor([parent], height_list=[height],
                                                compare_height=0.01, new_name_for_object="Surface")[0]
                else:
                    print(f"The surface is going to be taken from {parent.get_name()}")
                    surface_obj = slice_faces_with_normals(parent) 
                
                was_putted_on = True

            elif relation_to_establish == 1: # If == 1 take the inner surface of the parent; INSIDE relation
                height = np.min(parent.get_bound_box()[:,2]) + parent_attributes["surface_distance"]
                
                print(f"The surface is going to be taken from {parent.get_name()}")
                
                surface_obj = extract_floor([parent], height_list=[height],
                                            compare_height=0.01, new_name_for_object="Surface")[0]
                was_putted_inside = True

            if surface_obj is None: # If a surface couldn't be found
                if verbose:
                        print("The surface couldn't be found")
                return category_counter
        
            # Try to place max_n_tries objects in or on the actual parent
            number_of_tries = np.random.randint(1, max_n_tries)
            for ii in range(number_of_tries):  


                if len(objects_boxes) / 8 >= max_n_obj:
                    if verbose:
                        print("The maximal amount of objects was reached")
                    break
                
                
                # Load the object, which should be sampled on the surface

                if prioritize_relations: # Force the participation of relation generation objects (Microwaves, Trays ...)
                    
                    if parent_attributes["name"] == False:
                        new_children_index = np.random.choice([0, 2])
                    else:
                        new_children_index = np.random.randint(0, len(children_attributes))

                else:
                    new_children_index = np.random.randint(0, len(children_attributes))

                if new_children_index not in dropped_objects_types:
                    child_attributes = children_attributes[new_children_index]
                    dropped_objects_types.append(new_children_index)
                else:
                    if verbose:
                        print("There is already an instance of this type of object")
                    continue
                # child_attributes = children_attributes[n]
                sampling_obj = load_obj(child_attributes["path"])
                actual_attribute = 0
                if verbose:
                    print("--------------------------------------------------------------")
                    
                    print(f"The actual parent object is {parent.get_name()}")
                    print(f"And the relation(s) that are going to be generate is/are {relation_to_establish}")
                    print(f"This is the child number {ii}")
                    print(f"The name of that child object is {sampling_obj[0].get_name()}")
                
                if child_attributes["components"]:

                    component = sampling_obj[1] 
                    sampling_obj = [sampling_obj[0]]

                # Compare the size of the surface with the base of the bounding box from the sampling_obj
                surface_area = calculate_area_of_surface(surface_obj)
                sampling_obj_area = calculate_area_of_surface(sampling_obj[0], y_vector=np.array([0,0,1]))

                if parent_attributes["name"] == False and surface_area < 0.140:
                    if verbose:
                        print("The base object surface is too small")
                    break

                if True or "microwave" in sampling_obj[0].get_name():
                    print()
                    print("===============================================================")
                    print(f"The base of the child object is {round(sampling_obj_area, 3)} m^2")
                    print(f"The area of the {parent.get_name()} surface is {round(surface_area, 3)} m^2")
                if verbose:
                    print(f"The base of the child object is {round(sampling_obj_area, 3)} m^2")
                    print(f"The area of the parent surface is {round(surface_area, 3)} m^2")

                if surface_area <= sampling_obj_area + 0.06 and parent_attributes["name"] != False:
                    sampling_obj[0].delete()
                    print("The area surface of the parent object is to small")

                    if verbose: 
                        print("--------------------------------------------------------------")
                    continue

                dropped_object = place_object(sampling_obj, surface_obj, room_objs, dropped_object_list, verbose=verbose)

                    

                if not dropped_object[0]: # If the sampling_obj couldn't be placed
                    if verbose:
                        print("sampling_obj couldn't be placed")
                        print("--------------------------------------------------------------")
                    continue

                if verbose: 
                    print("--------------------------------------------------------------")

                # Store the type of relation of the dropped object
                if parent_attributes["name"] != include_base_object:

                    parent_name = "".join(parent.get_name())
                    child_name = "".join(dropped_object[0].get_name())

                    if was_putted_on:
                        relation = f"{child_name} ON {parent_name}"
                    elif was_putted_inside:
                        relation = f"{child_name} INSIDE {parent_name}"

                    relations_and_attributes["relation"].append(relation)

                
                # Store the bounding boxes of the objects to calculate their size and location later
                objects_boxes.extend(dropped_object[0].get_bound_box())

                # Prepare the next child to place
                # next_child = children_attributes[np.random.randint(0, len(children_attributes))]

                if child_attributes["components"] == "door":

                    actual_attribute = door_sampling(sampling_obj[0], component, room_objs, dropped_object_list, bvh_cache)
                    # if door_rotation < np.pi/4:
                    #     actual_attribute = 1.0 # The door is open

                    child_attributes["tags"][1] = actual_attribute

                category_counter = sample_scene_graph(sampling_obj[0], child_attributes, children_attributes, objects_boxes, 
                                                    dropped_object_list, category_counter, bvh_cache, room_objs, 
                                                    relations_and_attributes, verbose=verbose, max_n_tries=max_n_tries,
                                                        max_n_obj=max_n_obj, dropped_objects_types=dropped_objects_types,
                                                        prioritize_relations=prioritize_relations) 
                
                # if child_attributes["components"] == "door":

                #     door_rotation = door_sampling(sampling_obj[0], component, room_objs, dropped_object_list, bvh_cache)
                #     if door_rotation < np.pi/4:
                #         actual_attribute = 1.0 # The door is open
                if child_attributes["components"] == "door":
                    sampling_obj[0].join_with_other_objects([component])

                # Set the custom property in the remaining objects of interest
                dropped_object[0].set_cp("category_id", category_counter + 1)
                category_counter += 1
                dropped_object_list.extend(dropped_object)

                
                # If the object has a special characteristic that needs to be described (microwave open)
                
                relations_and_attributes["attribute"].append(int(actual_attribute))
            was_putted_inside = False
            was_putted_on = False
            # join surface objects again
            parent.join_with_other_objects([surface_obj])

                
    
    return  category_counter


def place_object(obj_to_place: list[MeshObject], surface_obj: MeshObject, room_objs: list[MeshObject], dropped_objects: list[MeshObject],
                 verbose=True):

    """ Function to place an object on other, considering that the base or receiving object could have 
     preestablished objects on it. This function places the object and tests if it is close enough to
     the surface.
      
    :param obj_to_place: New object to place in a list.
    :param surface_obj: Surface on which the new object is to be placed
    :param room_objs: Objects in the loaded room to check collisions with.
    :param dropped_objects: New objects that have been placed in the loaded room.

    :return: Type: :class:`List[MeshObject]`, list of placed objects or None when no object could be placed"""

    surface_height_z = np.max(surface_obj.get_bound_box(), axis=0)[2]

    def sample_pose(obj2: MeshObject):
        # Sample the objects location above the surface
        obj2.set_location(upper_region(
            objects_to_sample_on=[surface_obj],
            min_height=0.01,
            max_height=1,
            use_ray_trace_check=False,
            face_sample_range=[0.1, 0.9]
        ))
        #Randomized rotation of the sampled object
        obj2.set_rotation_euler(uniformSO3(around_y=False, around_x=False, around_z=True))

    objects_to_check = room_objs + dropped_objects
    
    # Sampling of the object
    if "mug" in obj_to_place[0].get_name():
        check_all_corners = False
    else:
        check_all_corners = True

    dropped_object_list_temp = sample_poses_on_surface(obj_to_place, surface_obj, sample_pose,
                                                       max_tries = 500, min_distance=0.1, max_distance=10,
                                                       check_all_bb_corners_over_surface=check_all_corners,
                                                       objects_to_check=objects_to_check, verbose=verbose )



    if not dropped_object_list_temp :
        
        print("Object couldn't be placed")
        return None, None
    



    # Enable physics for objects of interest (active) and the surface (passive)
    # If the object already has the rigid body features enabled, disable those and set the desired behavior
    
    
    dropped_object_list_temp[0].enable_rigidbody(True)

    surface_obj.enable_rigidbody(False)


    # Run the physics simulation
    simulate_physics_and_fix_final_poses(min_simulation_time=2, max_simulation_time=4,
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



def calculate_area_of_surface(object_to_measure: MeshObject, x_vector=np.array([1,0,0]), y_vector=np.array([0,1,0])):

    """ Calculate the area of a surface.

    :param object_to_measure: Surface with the area of interest.
    :param x_vector: Direction of the x axis.
    :param y_vector: Direction of the y axis.

    :return: Type: :class:`float`, Calculated area.
       """

    box = object_to_measure.get_bound_box()

    x_length = max(x_vector.dot(corner) for corner in box) - min(x_vector.dot(corner) for corner in box)
    y_length = max(y_vector.dot(corner) for corner in box) - min(y_vector.dot(corner) for corner in box)

    return x_length * y_length

def door_sampling(base: MeshObject, door: MeshObject, room_objs: list[MeshObject], placed_objects: list[MeshObject],
                  bvh_cache: Optional[Dict[str, mathutils.bvhtree.BVHTree]]):
    
    """ Samples the door of objects with that component with a randomized aperture.
    
    :param base: Object which has a door.
    :param door: MeshObject of the door.
    :param room_objs: Objects in the loaded room to check collisions with.
    :param placed_objects: New objects that have been placed in the loaded room.
    :param bvh_cache: The bvh_cache adds all current objects to the bvh tree, which increases the speed."""

    door.set_location( base.get_origin() )

    for i in range(10):
        door
        rotation = np.random.uniform(- np.pi/10, np.pi/10)
        door.set_rotation_euler(base.get_rotation() + [0, 0, rotation]) # from 0 to pi/2.5
        if CollisionUtility.check_intersections(door, bvh_cache, room_objs + placed_objects, []):
            return True # Open

    rotation = np.pi/2
    door.set_rotation_euler(base.get_rotation() + [0, 0, rotation])

    return False # close