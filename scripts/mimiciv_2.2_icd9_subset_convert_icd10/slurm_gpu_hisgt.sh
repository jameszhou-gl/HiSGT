#!/bin/bash --login
# ! specify job-name, gpu (l40, a100, h100)
#SBATCH --nodes=1
#SBATCH --job-name=mimiciv_hisgt
#SBATCH -o output/slurm-%j.output
#SBATCH -e output/slurm-%j.error
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --account=a_barbieri

# Load environment and activate conda
module load anaconda3/2022.05
conda activate hisgt
# Configuration
dataset="mimiciv"
dataset_version="2.2_icd9_subset_convert_icd10"

TIMESTAMP=$(date '+%Y-%m-%d_%H_%M_%S')
BASE_LOG_DIR="/home/uqgzhou1/code/HiSGT/output/$TIMESTAMP"
mkdir -p "$BASE_LOG_DIR"

# Save script and Git info
cp "$0" "$BASE_LOG_DIR/slurm_script.sh"
{
    echo "Git Commit ID: $(git rev-parse HEAD)"
    echo "Git Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Git Tag: $(git describe --tags --exact-match HEAD 2>/dev/null || echo 'No tag')"
    echo "Git Remote URL: $(git remote get-url origin)"
    echo -e "Git Status:\n$(git status --short)"
} > "$BASE_LOG_DIR/git_info.log"

# GPU info
echo "Logging GPU information"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader,nounits > "$BASE_LOG_DIR/gpu_info.log"

echo "Job started at: $(date '+%Y-%m-%d-%H_%M_%S')"

beta=0.8
gamma=0.5

# Create a unique log directory for each run
LOG_DIR="$BASE_LOG_DIR/${dataset}_${dataset_version}_run_hisgt_beta_${beta}_gamma_${gamma}"
mkdir -p "$LOG_DIR"
echo "Running HiSGT (Run $i) on $dataset version $version with beta=$beta and gamma=$gamma..."

# Run the training script
srun python baselines/hisgt/train_hisgt.py --LOG_DIR $LOG_DIR \
    --dataset $dataset \
    --dataset_version $dataset_version \
    --total_vocab_size 9102 \
    --code_vocab_size 9072 \
    --label_vocab_size 25 \
    --special_vocab_size 5 \
    --n_ctx 864 \
    --n_positions 914 \
    --dropout 0.1 \
    --n_layer 6 --n_head 8 --n_embd 384 \
    --epoch 100 \
    --flag_vec_semantic \
    --flag_vec_hierarchy \
    --beta $beta \
    --gamma $gamma

# Run the sampling script
srun python baselines/hisgt/sample_hisgt.py --LOG_DIR $LOG_DIR \
    --dataset $dataset \
    --dataset_version $dataset_version \
    --dataset $dataset \
    --dataset_version $dataset_version \
    --total_vocab_size 9102 \
    --code_vocab_size 9072 \
    --label_vocab_size 25 \
    --special_vocab_size 5 \
    --n_ctx 864 \
    --n_positions 914 \
    --dropout 0.1 \
    --n_layer 6 --n_head 8 --n_embd 384 \
    --epoch 100 \
    --flag_vec_semantic \
    --flag_vec_hierarchy \
    --beta $beta \
    --gamma $gamma

# Run the evaluation script
python evaluation/run_evaluation.py --output_path $LOG_DIR \
    --dataset $dataset \
    --dataset_version $dataset_version

# Cleanup unnecessary files
find "$LOG_DIR/utility_models" -type f -name "*.pt" -delete
echo "Deleted .pt files in $LOG_DIR/utility_models"

# Finalize
echo "Job finished at: $(date '+%Y-%m-%d-%H_%M_%S')"
mv "output/slurm-${SLURM_JOB_ID}.output" "$BASE_LOG_DIR/slurm-${SLURM_JOB_ID}.output"
mv "output/slurm-${SLURM_JOB_ID}.error" "$BASE_LOG_DIR/slurm-${SLURM_JOB_ID}.error"

