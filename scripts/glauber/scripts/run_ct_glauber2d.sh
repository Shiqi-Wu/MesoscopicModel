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
T_FRAC=3
T_END=5.0
SNAPSHOT_DT=0.1
ROUNDS=1
BLOCK=8
KERNEL="nearest"  # "nearest" or "gaussian"

OUTDIR="results/ct_glauber"

python scripts/glauber/tasks/main_ct_glauber2d.py \
    --size $SIZE \
    --ell $ELL \
    --sigma $SIGMA \
    --tau $TAU \
    --m0 $M0 \
    --seed $SEED \
    --J $J \
    --h $H \
    --T_frac $T_FRAC \
    --t_end $T_END \
    --snapshot_dt $SNAPSHOT_DT \
    --rounds $ROUNDS \
    --block $BLOCK \
    --outdir $OUTDIR \
    --kernel $KERNEL \

