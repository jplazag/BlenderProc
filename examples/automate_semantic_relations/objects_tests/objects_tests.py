import blenderproc as bproc
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('treed_obj_path', help="Path to the downloaded 3D object")
parser.add_argument('output_dir', nargs='?', default="examples/automate_semantic_relations/Test_01/output", help="Path to where the final files, will be saved")
args = parser.parse_args()



def calculate_area_of_surface(surface: bproc.types.MeshObject):
    box = surface.get_bound_box()

    z = box[0,2]
    y = box[0,1]
    x = box[0,0]

    print(box[1:,:])
    z1 = 0
    Y = False
    X = False
    print("--------------------------")
    for point in box[1:,:]:
        
        if point[2] == z:
            print(point)
            if point[1] == y and not X:
                x1 = point[0]
                X = True
            if point[0] == x and not Y:
                y1 = point[1]
                Y = True
                # print(y, y1)
        else:
            z1 = point[2]
    print("--------------------------")
    return (y1 - y) * (x1 - x)*1000, (y1 - y) * (x1 - x) * (z1 - z) 


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

print(surface.get_bound_box_volume())
print(obj[0].get_bound_box())
print(surface.get_bound_box())
print(calculate_area_of_surface(surface))

min = np.max(np.linalg.norm(  obj[1].get_bound_box(), axis=0 ) )



for point in obj[1].get_bound_box():

    if np.linalg.norm(point) < min:
        min = np.linalg.norm(point)
        min_point = point

    

obj[1].set_origin( min_point )
obj[1].set_rotation_euler(obj[1].get_rotation() - [0, np.pi/3, 0]) # from 0 to pi/2.5



locations = [
    [5.8, -1.0, 0.0],
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