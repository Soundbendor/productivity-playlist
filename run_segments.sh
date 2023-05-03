#!/bin/bash

#SBATCH -J playlist-segments      #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH -c 6
#SBATCH -o ./log/test_segments.out
#SBATCH -e ./log/test_segments.err
#SBATCH -t 4-00:00:00      #set max job time to 4 days, 0h (default is around 36-48h)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gaurs@oregonstate.edu

#my commands
source ./venv/bin/activate      #activate tensorflow environment
#source activate tf_gpu
# python3 test_segments.py 100
python3 analyze_tests_datasets.py segment
deactivate

