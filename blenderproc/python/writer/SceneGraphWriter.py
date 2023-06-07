import numpy as np
import cv2
from typing import Optional, Dict, Union, Tuple, List
import h5py
import csv
import bpy
import os

from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.Utility import Utility

def write_scene_graph(output_dir, h5_file_path, scene_objects: list[MeshObject], data, relations_and_features: Dict, 
                      camera_counter: int, test_bboxes: bool = False, relations_number: int = 2 ):
    
    """ Function that writes the annotations of a scene graph. It generates a h5 file with the image of 
     the graph, the bounding boxes of each element in the frame, the name of the objects and their relations
      with each other.
    :param h5_file_path: Path in which the annotations are going to be stored.
    :param scene_objects: Objects detected in the current scene.
    :param data: Variable that contains the information about the images and the segmentation of the objects.
    :param relations_and_features: Dictionary with the relations of each child object and its special features or attributes
                                    like open or close for the microwave.
    :param camera_counter: Counter of the stored camera poses, which represents the number of frames.
    :param test_bboxes: If True, run a routine that prints the images with the bounding boxes.
    :param relations_number: Number of considered relations betwen objects, as default 2 (ON and INSIDE)."""
    
    annotations_file = h5py.File(os.path.join(output_dir,h5_file_path), 'a',track_order=True)

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
        
        # Get bounding boxes of every object in the current frame
        bboxes = bbox_from_segmented_images(instance_segmaps=data["instance_segmaps"], 
                                            instance_attribute_maps=data["instance_attribute_maps"],
                                            colors=data["colors"])
        
        if test_bboxes:
            test_bounding_boxes(data['colors'][rendered_image], bboxes[rendered_image], scene_number + rendered_image,
                                output_dir)

        annotations_file[group_name].create_dataset('bboxes', data=np.array(bboxes[rendered_image]))

        annotations_file[group_name].create_dataset('image', data=np.array(data['colors'][rendered_image])) 

        annotations_file[group_name].create_dataset('image_seg', data=np.array(data['instance_segmaps'][rendered_image])) 

        annotations_file[group_name].create_dataset('objects', (objects_number,), data=np.array(objects))

        annotations_file[group_name].create_dataset('relations', data=np.array(relations))
    
    annotations_file.close()


def bbox_from_segmented_images(instance_segmaps: Optional[List[np.ndarray]] = None, instance_attribute_maps: Optional[List[dict]] = None, 
                               colors: Optional[List[np.ndarray]] = None, segmap_output_key: str = "segmap", segcolormap_output_key: str = "segcolormap", 
                               rgb_output_key: str = "colors"):
    
    """ Function that takes the segmentated images and generates bounding boxes for each object on the scene. """

    instance_segmaps = [] if instance_segmaps is None else list(instance_segmaps)
    colors = [] if colors is None else list(colors)
    if instance_attribute_maps is None:
        instance_attribute_maps = []

    if len(colors) > 0 and len(colors[0].shape) == 4:
        raise ValueError("BlenderProc currently does not support writing coco annotations for stereo images. "
                            "However, you can enter left and right images / segmaps separately.")

    if not instance_segmaps:
        # Find path pattern of segmentation images
        segmentation_map_output = Utility.find_registered_output_by_key(segmap_output_key)
        if segmentation_map_output is None:
            raise RuntimeError(f"There is no output registered with key {segmap_output_key}. Are you sure you "
                                f"ran the SegMapRenderer module before?")

    if not colors:
        # Find path pattern of rgb images
        rgb_output = Utility.find_registered_output_by_key(rgb_output_key)
        if rgb_output is None:
            raise RuntimeError(f"There is no output registered with key {rgb_output_key}. Are you sure you "
                                f"ran the RgbRenderer module before?")

    if not instance_attribute_maps:
        # Find path of name class mapping csv file
        segcolormap_output = Utility.find_registered_output_by_key(segcolormap_output_key)
        if segcolormap_output is None:
            raise RuntimeError(f"There is no output registered with key {segcolormap_output_key}. Are you sure you "
                                f"ran the SegMapRenderer module with 'map_by' set to 'instance' before?")
        
    # collect all mappings from csv (backwards compat)
    segcolormaps = []
    # collect all instance segmaps (backwards compat)
    inst_segmaps = []

    # for each rendered frame
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):

        if not instance_attribute_maps:
            # read colormappings, which include object name/class to integer mapping
            segcolormap = []
            with open(segcolormap_output["path"] % frame, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for mapping in reader:
                    segcolormap.append(mapping)
            segcolormaps.append(segcolormap)

        if not instance_segmaps:
            # Load segmaps (backwards compat)
            segmap = np.load(segmentation_map_output["path"] % frame)
            inst_channel = int(segcolormap[0]['channel_instance'])
            inst_segmaps.append(segmap[:, :, inst_channel])

    instance_attribute_maps = segcolormaps if segcolormaps else instance_attribute_maps
    instance_segmaps = inst_segmaps if inst_segmaps else instance_segmaps
    
    instance_2_category_maps = []

    for inst_attribute_map in instance_attribute_maps:
        instance_2_category_map = {}
        for inst in inst_attribute_map:
            # skip background
            if int(inst["category_id"]) != 0:
                
                instance_2_category_map[int(inst["idx"])] = int(inst["category_id"])

        instance_2_category_maps.append(instance_2_category_map)

    bounding_boxes_per_inst_segmap = []

    for inst_segmap, instance_2_category_map in zip(instance_segmaps, instance_2_category_maps):
        bounding_boxes = []

        # Go through all objects visible in this image
        instances = np.unique(inst_segmap)
        # Remove background
        instances = np.delete(instances, np.where(instances == 0))
        for inst in instances:
            if inst in instance_2_category_map:
                # Calc object mask
                binary_inst_mask = np.where(inst_segmap == inst, 1, 0)
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