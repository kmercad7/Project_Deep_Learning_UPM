#!/bin/bash

set -euo pipefail

DATASET=OPP

for FOLD in 1 2 3 4; do
    echo "Submitting pretraining for fold ${FOLD}"

    PRE_ID=$(sbatch --parsable pretrain_OPP_mamba.slurm "$DATASET" "$FOLD")
    echo "Pretraining job for fold ${FOLD}: ${PRE_ID}"
    echo "Submitting downstream dependent on ${PRE_ID}"

    DOWN_ID=$(sbatch --parsable --dependency=afterok:${PRE_ID} train_downstream_OPP_mamba.slurm "$DATASET" "$FOLD")
    echo "Downstream job ID for fold ${FOLD}: ${DOWN_ID}"
done
