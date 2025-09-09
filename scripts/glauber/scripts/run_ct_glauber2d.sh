#!/bin/bash

# Initialization parameters
ELL=32.0
SIGMA=1.2
TAU=1.0
M0=0.1
SEED=0

# Simulation parameters
SIZE=1024
J=1.0
H=0.0
T=4.0
T_END=5.0
SNAPSHOT_DT=0.01
ROUNDS=1
BLOCK=8
KERNEL="gaussian"  # "nearest" or "gaussian"
# EPISILON=0.015625  # only used if KERNEL is "gaussian"
EPISILON=0.03125  # only used if KERNEL is "gaussian"

# KERNEL="nearest"  # "nearest" or "gaussian"
# EPISILON=0.0  # only used if KERNEL is "gaussian

OUTDIR="data/ct_glauber_2"

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
    --kernel $KERNEL \
    --epsilon $EPISILON \
    --use_gpu \

