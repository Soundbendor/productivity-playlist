#!/bin/bash

#SBATCH -J playlist-analyze      #job name
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH -c 6
#SBATCH -o ./log/run_analyze.out
#SBATCH -e ./log/run_analyze.err
#SBATCH -t 4-00:00:00      #set max job time to 4 days, 0h (default is around 36-48h)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gaurs@oregonstate.edu

#my commands
source ./venv/bin/activate      #activate tensorflow environment
#source activate tf_gpu
python3 analyze_tests.py dataset
python3 analyze_tests.py distance
python3 analyze_tests.py length
python3 analyze_tests.py kval
python3 analyze_tests.py segment
deactivate

