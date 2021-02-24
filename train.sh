#!/bin/bash

#SBATCH -J playlist      #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH -o train.out
#SBATCH -e train.err
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn-m-1
#SBATCH -t 4-00:00:00      #set max job time to 4 days, 0h (default is around 36-48h)

#my commands
source ./venv/bin/activate      #activate tensorflow environment
python3 model.py
deactivate