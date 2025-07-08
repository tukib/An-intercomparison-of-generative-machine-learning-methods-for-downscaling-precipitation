#!/bin/bash -l
#SBATCH --partition=hgx
#SBATCH --time=71:59:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=A100:1
#SBATCH --account=niwa03712
#SBATCH --mail-user=bwar780@aucklanduni.ac.nz
#SBATCH --mail-type=ALL
#SBATCH --output log/%j-%x.out
#SBATCH --error log/%j-%x.out

module purge
module load NeSI
module load TensorFlow/2.13.0-gimkl-2022a-Python-3.11.3
export PYTHONNOUSERSITE=1
source /nesi/nobackup/niwa00018/bwar780/tf_venv/bin/activate

nvidia-smi

# touch "/nesi/project/niwa00018/bwar780/multi-variate-gan/.${SLURM_JOB_NAME}.job.active"

python /nesi/project/niwa00018/bwar780/multi-variate-gan/train_model_rain_future_updated_diffusion.py $1

# rm -f "/nesi/project/niwa00018/bwar780/multi-variate-gan/.${SLURM_JOB_NAME}.job.active"

### README:
# sbatch -J <jobname> run_diffusion_scripts.sl <config>
