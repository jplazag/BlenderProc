#!/bin/sh

source /home/plaz_ju/mambaforge/bin/activate

conda activate /net/rmc-lx0351/home_local/plaz_ju/mohits_home_folder/mambaforge/envs/sornet/

# script=/net/rmc-lx0351/home_local/plaz_ju/mohits_home_folder/sornet/train_clevr.py
# data_dir=/net/rmc-lx0351/home_local/plaz_ju/mohits_home_folder/Deformable-DETR/data/sgg_task_1/annotations
# log_dir=/net/rmc-lx0351/home_local/plaz_ju/mohits_home_folder/sornet/log/blender_proc_log/mohits_data

# python3 $script --data_dir $data_dir --log_dir $log_dir --max_nobj 3 --batch_size 32

script=/net/rmc-lx0351/home_local/plaz_ju/mohits_home_folder/sornet/train_clevr.py
data_dir=/home/plaz_ju/Documents/BlenderProc/examples/advanced/automate_semantic_relations/annotations/one_instance
log_dir=/net/rmc-lx0351/home_local/plaz_ju/mohits_home_folder/sornet/log/blender_proc_log/one_instance

python3 $script --data_dir $data_dir --log_dir $log_dir --max_nobj 4 --batch_size 32

