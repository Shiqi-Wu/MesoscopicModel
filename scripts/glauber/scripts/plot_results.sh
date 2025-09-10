# This script is to plot the results from the CT Glauber simulation

#!/bin/bash

FOLDER="data/ct_glauber_2/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_T1_J1_h0_tend5.0_dt0.01_block8_kernelgaussian_epsilon0.015625_seed0/round0"

python scripts/glauber/tasks/main_plot_results.py $FOLDER