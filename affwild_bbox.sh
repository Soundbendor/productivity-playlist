#!/bin/bash

#SBATCH -J affwild_bbox    #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH -o affwild_bbox.out
#SBATCH -e affwild_bbox.err
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00      #set max job time to 4 days, 0h (default is around 36-48h)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gaurs@oregonstate.edu

#my commands
#source ./venv/bin/activate      #activate tensorflow environment
source activate tf_gpu
#cd data/affwild
#rm -rf train
#cd ./bboxes
#tar -xvf train.tar.gz
#cd ../landmarks
#tar -xvf train.tar.gz
python3 affwild_bbox.py
#deactivate

