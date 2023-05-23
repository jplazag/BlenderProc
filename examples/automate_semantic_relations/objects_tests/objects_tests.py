import blenderproc as bproc
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('treed_obj_path', help="Path to the downloaded 3D object")
parser.add_argument('output_dir', nargs='?', default="examples/automate_semantic_relations/Test_01/output", help="Path to where the final files, will be saved")
args = parser.parse_args()



def calculate_area_of_surface(object_to_measure: bproc.types.MeshObject, x_vector=np.array([1,0,0]), y_vector=np.array([0,1,0])):

    box = object_to_measure.get_bound_box()

    x_lenght = max(x_vector.dot(corner) for corner in box) - min(x_vector.dot(corner) for corner in box)
    y_lenght = max(y_vector.dot(corner) for corner in box) - min(y_vector.dot(corner) for corner in box)
    return x_lenght * y_lenght


bproc.init()


# set the light bounces
light = bproc.types.Light()
light.set_type("POINT")

light.set_location([1.8, -2.0, 0.0])



light.set_energy(50)

bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

# define the camera intrinsics
bproc.camera.set_resolution(512, 512)


# Load the object, which should be sampled on the surface
obj = bproc.loader.load_obj(args.treed_obj_path)


obj_size = np.max(np.max(obj[0].get_bound_box(), axis=0) - np.min(obj[0].get_bound_box(), axis=0))

# for x in obj:
#     print(bproc.object.slice_faces_with_normals(x).get_bound_box())

surface = bproc.object.slice_faces_with_normals(obj[0])

surface2 = bproc.object.extract_floor([obj[0]], compare_height=0.1)[0]

# surface.set_location(surface.get_location() + [0, 0, 0.2])

# surface2[0].set_location(surface2[0].get_location() + [0, 0, -0.2])

# print(surface2)

# print(surface.get_bound_box_volume())
# print(obj[0].get_bound_box())
# print(surface.get_bound_box())
# print(calculate_area_of_surface(surface))

# obj[0].join_with_other_objects([obj[1]])

# obj[1].delete()

# obj = [obj[0]]

print(calculate_area_of_surface(surface))

print(calculate_area_of_surface(surface, x_vector=np.array([1,0,0]), y_vector=np.array([0,0,1])))

print(calculate_area_of_surface(surface, x_vector=np.array([0,1,0]), y_vector=np.array([0,0,1])))


print(calculate_area_of_surface(surface2))

print(calculate_area_of_surface(surface2, x_vector=np.array([1,0,0]), y_vector=np.array([0,0,1])))

print(calculate_area_of_surface(surface2, x_vector=np.array([0,1,0]), y_vector=np.array([0,0,1])))


print(calculate_area_of_surface(obj[0]))

print(calculate_area_of_surface(obj[0], x_vector=np.array([1,0,0]), y_vector=np.array([0,0,1])))

print(calculate_area_of_surface(obj[0], x_vector=np.array([0,1,0]), y_vector=np.array([0,0,1])))

""" min = np.max(np.linalg.norm(  obj[1].get_bound_box(), axis=0 ) )



for point in obj[1].get_bound_box():

    if np.linalg.norm(point) < min:
        min = np.linalg.norm(point)
        min_point = point

    

obj[1].set_origin( min_point )
obj[1].set_rotation_euler(obj[1].get_rotation() - [0, 0*np.pi/3, 0]) # from 0 to pi/2.5 """



locations = [
    [1.8, -0.5, 1.0],
    [1.8, -0.5,-1.0],
    [1.8, -0.5, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, -1.0, 0.0],
    [0.0, -1.0, 0.0]
]

poses = 0

for location in locations:

    poi = bproc.object.compute_poi(obj)

    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.pi/2)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    
    bproc.camera.add_camera_pose(cam2world_matrix,
                                    frame = poses)
    poses += 1

data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=False)