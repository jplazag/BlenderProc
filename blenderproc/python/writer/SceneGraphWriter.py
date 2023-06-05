import numpy as np
from typing import Dict
import h5py

from blenderproc.python.types.MeshObjectUtility import MeshObject

def write_scene_graph(h5_file_path, scene_objects: list[MeshObject], data, relations_and_features: Dict, 
                      camera_counter: int, bboxes: list, relations_number: int = 2 ):
    
    annotations_file = h5py.File(h5_file_path, 'a',track_order=True)

    if list(annotations_file.keys()):

        scene_number = int(list(annotations_file["/"].keys())[-1]) + 1 
    else:
        scene_number = 0


    objects_number = len(scene_objects)
    relations = np.zeros(shape=(relations_number, objects_number, objects_number))
    
    for r_n in range(relations_number):
        np.fill_diagonal(relations[r_n,:,:], -1)

    objects = [scene_object.get_name().encode('utf-8') for scene_object in scene_objects]

    
    for counter, scene_object in enumerate(scene_objects):

        current_relation = relations_and_features["relation"][counter].split()

        # Since "NONE" relations does not have a related object, it is not required to search for that object's name

        if current_relation[0] != "NONE":
            parent_index = objects.index(" ".join(current_relation[1:]).encode('utf-8'))
            child_index = objects.index(scene_object.get_name().encode('utf-8'))

            if current_relation[0] == "ON":
                
                relations[0, child_index, parent_index] = 1
            elif current_relation[0] == "INSIDE":
                
                relations[1, child_index, parent_index] = 1

    for rendered_image in range(camera_counter):
        group_name = str(scene_number + rendered_image)
        annotations_file.create_group(group_name, track_order=True)
        annotations_file[group_name].create_dataset('attributes', data=np.array([relations_and_features["attribute"]]))
        
        annotations_file[group_name].create_dataset('bboxes', data=np.array(bboxes[rendered_image]))

        annotations_file[group_name].create_dataset('image', data=np.array(data['colors'][rendered_image])) 
        annotations_file[group_name].create_dataset('image_seg', data=np.array(data['instance_segmaps'][rendered_image])) 
        annotations_file[group_name].create_dataset('objects', (objects_number,), data=np.array(objects))

        annotations_file[group_name].create_dataset('relations', data=np.array(relations))
    
    annotations_file.close()