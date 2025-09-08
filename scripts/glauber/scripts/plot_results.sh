# This script is to plot the results from the CT Glauber simulation

#!/bin/bash

FOLDER="/home/shiqi_w/code/MesoscopicModel/results/ct_glauber/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0/round0"

python scripts/glauber/tasks/main_plot_results.py $FOLDER