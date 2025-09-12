# This script is to plot the results from the CT Glauber simulation

#!/bin/bash

FOLDER="data/ct_glauber/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_T1_J1_h0_tend20.0_dt0.01_block8_kernelgaussian_epsilon0.03125_seed0/round0"

python scripts/glauber/tasks/main_plot_results.py $FOLDER