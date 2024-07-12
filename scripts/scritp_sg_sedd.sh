#!/bin/bash -l

#SBATCH -p gpuxl
#SBATCH -C h100
#SBATCH -J "mdlm"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --open-mode=append     
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --output=logs/run-%j.log
#SBATCH --time=48:00:00

export HF_HOME=/mnt/home/sgolkar/hfcache
export TRANSFORMERS_CACHE=/mnt/home/sgolkar/trans_cache

source ~/envs/mdlm/bin/activate

srun python -u -m main \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  model=small \
  data=coconut-split \
  wandb.name=sedd-split-run2 \
  parameterization=sedd \
  model.length=120 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  sampling.predictor=analytic \
  time_conditioning=True