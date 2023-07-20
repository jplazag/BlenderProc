#!/bin/bash


source /home/plaz_ju/mambaforge/bin/activate

conda activate blenderproc

count=0

cli=/home/plaz_ju/Documents/BlenderProc/cli.py
script="/home/plaz_ju/Documents/BlenderProc/examples/advanced/automate_semantic_relations/main.py"
scene="/volume/reconstruction_data/datasets/3d_front/release/3D-FRONT/10087356-5564-4bb8-9dcd-72adee753630.json"
model="/volume/reconstruction_data/datasets/3d_front/release/3D-FUTURE-model"
texture="/volume/reconstruction_data/datasets/3d_front/release/3D-FRONT-texture"
odb_objects="/net/rmc-lx0351/home_local/plaz_ju/odb"

# output_path="/home/plaz_ju/Documents/BlenderProc/examples/advanced/automate_semantic_relations/annotations/"
# output_path="/home_local/plaz_ju/generated_data/annotations/"
# output_directory="one_instance100/"


# output_path="/net/rmc-gpu16/home_local/plaz_ju/"



# Data generation focusing the objects (avoid the small offset used to generate more diverse data and not allways the camera aiming to the objects)


output_path="/home_local/plaz_ju/"
output_directory="focused/"

extension=".h5"

output_file="training"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 1200 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension --objects_focused 
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done

output_file="validation"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 200 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension --objects_focused 
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done

output_file="test"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 200 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension --objects_focused 
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done



# Generate a dataset prioritizing the relation for the 42% of the training dataset

output_path="/home_local/plaz_ju/"
output_directory="one_instance_100/"

extension=".h5"

output_file="training"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 500 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension --prioritize_relations 
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done

output_file="validation"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 50 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension --prioritize_relations 
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done

output_file="test"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 50 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension --prioritize_relations 
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done










output_file="training"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 1200 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension 
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done

output_file="validation"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 200 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension 
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done

output_file="test"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 200 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension 
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done


# Data generation prioritizing the creation of relations between objects


output_directory="prioritize_relations/"

extension=".h5"

output_file="training"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 1200 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension --prioritize_relations
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done

output_file="validation"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 200 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension --prioritize_relations
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done

output_file="test"

count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)

while [ $count -lt 200 ]
do
    python3 $cli run $script $scene $model $texture $odb_objects $output_path$output_directory$output_file $output_file$extension --prioritize_relations
    count=$(find $output_path$output_directory$output_file -maxdepth 1 -type f -name "*.hdf5" | wc -l)
done