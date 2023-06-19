import numpy as np
import cv2
from typing import Optional, Dict, Union, Tuple, List
import h5py
import csv
import bpy
import os

from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.Utility import Utility

def write_scene_graph(output_dir, h5_file_name, objects_on_frames: list[MeshObject], data, 
                      relations_and_features: Dict, relations_number: int = 2 ):
    
    """ Function that writes the annotations of a scene graph. It generates a h5 file with the image of 
     the graph, the bounding boxes of each element in the frame, the name of the objects and their relations
      with each other.
    :param h5_file_name: Path in which the annotations are going to be stored.
    :param objects_on_frames: Objects detected in the current frames.
    :param data: Variable that contains the information about the images and the segmentation of the objects.
    :param relations_and_features: Dictionary with the relations of each child object and its special features or attributes
                                    like open or close for the microwave.
    :param test_bboxes: If True, run a routine that prints the images with the bounding boxes.
    :param relations_number: Number of considered relations betwen objects, as default 2 (ON and INSIDE)."""
    
    annotations_file = h5py.File(os.path.join(output_dir,h5_file_name), 'a',track_order=True)

    # Get bounding boxes of every object in the current frame
    bboxes = bbox_from_segmented_images(instance_segmaps=data["instance_segmaps"], 
                                        instance_attribute_maps=data["instance_attribute_maps"])

    if list(annotations_file.keys()):

        scene_number = int(list(annotations_file["/"].keys())[-1]) + 1 
    else:
        scene_number = 0
    
    
    for rendered_image in range(len(data['colors'])):

        objects = [scene_object.get_name().encode('utf-8') for scene_object in objects_on_frames[rendered_image]]
        
        objects_number = len(objects)
        relations = np.zeros(shape=(relations_number, objects_number, objects_number))
        
        for r_n in range(relations_number):
            np.fill_diagonal(relations[r_n,:,:], -1)

        for counter, scene_object in enumerate(objects_on_frames[rendered_image]):

            current_relation = relations_and_features["relation"][counter].split()

            # Since "NONE" relations does not have a related object, it is not required to search for that object's name

            if current_relation[0] != "NONE":
                parent_index = objects.index(" ".join(current_relation[1:]).encode('utf-8'))
                child_index = objects.index(scene_object.get_name().encode('utf-8'))

                if current_relation[0] == "ON":
                    
                    relations[0, child_index, parent_index] = 1
                elif current_relation[0] == "INSIDE":
                    
                    relations[1, child_index, parent_index] = 1
            

        group_name = str(scene_number + rendered_image)
        annotations_file.create_group(group_name, track_order=True)
        annotations_file[group_name].create_dataset('attributes', data=np.array([relations_and_features["attribute"]]))
        
        
        
        # test_bounding_boxes(data['colors'][rendered_image], bboxes[rendered_image], scene_number + rendered_image,
        #                         output_dir)

        annotations_file[group_name].create_dataset('bboxes', data=np.array(bboxes[rendered_image]))

        annotations_file[group_name].create_dataset('image', data=np.array(data['colors'][rendered_image])) 

        annotations_file[group_name].create_dataset('image_seg', data=np.array(data['instance_segmaps'][rendered_image])) 

        annotations_file[group_name].create_dataset('objects', (objects_number,), data=np.array(objects))

        annotations_file[group_name].create_dataset('relations', data=np.array(relations))
    
    annotations_file.close()


def bbox_from_segmented_images(instance_segmaps: Optional[List[np.ndarray]] = None, instance_attribute_maps: Optional[List[dict]] = None):
    
    """ Function that takes the segmentated images and generates bounding boxes for each object on the scene. 
    :param instance_segmaps: List of images represented by ndarrays where each instanced object is represented with a different color,
                            so it is easy differentiable and thus segmented.
    :param instance_attribute_maps: List of Dictionaries that stores the instance and attributes values for each pixel, it relates
                                    the instance of each object on scene with its attributes (in this case its categry ID)"""
    
    instance_2_category_maps = []
    
    for inst_attribute_map in instance_attribute_maps:
        instance_2_category_map = {}
        for inst in inst_attribute_map:
            # skip background
            if int(inst["category_id"]) != 0:
                
                instance_2_category_map[int(inst["idx"])] = int(inst["category_id"])

        instance_2_category_maps.append(instance_2_category_map)

    bounding_boxes_per_inst_segmap = []
    
    for instance_segmaps, instance_2_category_map in zip(instance_segmaps, instance_2_category_maps):
        bounding_boxes = []


        # Go through all objects visible in this image
        instances = np.unique(instance_segmaps)
        # Remove background
        instances = np.delete(instances, np.where(instances == 0))
        for inst in instances:
            if inst in instance_2_category_map:
                # Calc object mask
                binary_inst_mask = np.where(instance_segmaps == inst, 1, 0)
                # Add coco info for object in this image
                rows = np.any(binary_inst_mask, axis=1)
                cols = np.any(binary_inst_mask, axis=0)
                # Find the min and max col/row index that contain 1s
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                # Calc height and width
                h = rmax - rmin + 1
                w = cmax - cmin + 1
                bounding_boxes.append([int(cmin), int(rmin), int(w), int(h)])
        bounding_boxes_per_inst_segmap.append(bounding_boxes)

    return bounding_boxes_per_inst_segmap

def test_bounding_boxes(color_image, bboxes, image_number, output_dir):
    
    for bbox in bboxes:
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]

        cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_dir,f'image_with_bboxes{image_number}.jpg'),color_image)      