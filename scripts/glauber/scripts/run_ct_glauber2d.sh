#!/bin/bash

# Initialization parameters
ELL=64.0
SIGMA=1.2
TAU=1.0
M0=0.1
SEED=0

# Simulation parameters
SIZE=2048
J=1.0
H=0.0
T=1.0
T_END=20.0
SNAPSHOT_DT=0.01
ROUNDS=20
BLOCK=16
KERNEL="gaussian"  # "nearest" or "gaussian"
# EPISILON=0.015625  # only used if KERNEL is "gaussian"
EPISILON=0.03125  # only used if KERNEL is "gaussian"
METHOD="tau-leaping"  # "gillespie" or "tau-leaping"
EPS_TAU=0.01  # only used if METHOD is "tau-leaping"


# KERNEL="nearest"  # "nearest" or "gaussian"
# EPISILON=0.0  # only used if KERNEL is "gaussian

OUTDIR="data/ct_glauber"
REMOTE_DIR="/mnt/nfs/homes/shiqi_w/MesoscopicModel/data/shared/ct_glauber"

python scripts/glauber/tasks/main_ct_glauber2d.py \
    --size $SIZE \
    --ell $ELL \
    --sigma $SIGMA \
    --tau $TAU \
    --m0 $M0 \
    --seed $SEED \
    --J $J \
    --h $H \
    --T $T \
    --t_end $T_END \
    --snapshot_dt $SNAPSHOT_DT \
    --rounds $ROUNDS \
    --block $BLOCK \
    --outdir $OUTDIR \
    --remote_dir $REMOTE_DIR \
    --kernel $KERNEL \
    --epsilon $EPISILON \
    --method $METHOD \
    --eps_tau $EPS_TAU \
    --use_gpu \

