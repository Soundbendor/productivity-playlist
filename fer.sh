#!/bin/bash

#SBATCH -J playlist      #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH -o ./out/fer.out
#SBATCH -e ./out/fer.err
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH -t 4-00:00:00      #set max job time to 4 days, 0h (default is around 36-48h)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gaurs@oregonstate.edu

#my commands
#source ./venv/bin/activate      #activate tensorflow environment
source activate tf_gpu
python3 model.py
#deactivate

