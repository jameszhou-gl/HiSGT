#!/bin/bash --login
# ! specify job-name, gpu (l40, a100, h100)
#SBATCH --nodes=1
#SBATCH --job-name=mimiciii_baselines
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

# ! Set the following flags to true to run the corresponding model
RUN_LSTM=true
RUN_GPT=true
# running problem in EVA
RUN_EVA=true
RUN_SynTEG=true

RUN_HALOCOARSE=true
RUN_HALO=true

# Configuration
dataset="mimiciii"
dataset_version="1.4_convert_icd10"

TIMESTAMP=$(date '+%Y-%m-%d_%H_%M_%S')
BASE_LOG_DIR="output/$TIMESTAMP"
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

# Train the model
run_lstm() {
    LOG_DIR="$BASE_LOG_DIR/${dataset}_${dataset_version}_lstm"
    mkdir -p "$LOG_DIR"
    echo "Running LSTM on $dataset version $version..."
    srun python baselines/lstm/train_lstm.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    srun python baselines/lstm/sample_lstm.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version 
    srun python evaluation/run_evaluation.py --output_path $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    find "$LOG_DIR/utility_models" -type f -name "*.pt" -delete
    echo "Deleted .pt files in $LOG_DIR/utility_models"
}

run_gpt() {
    LOG_DIR="$BASE_LOG_DIR/${dataset}_${dataset_version}_gpt"
    mkdir -p "$LOG_DIR"
    echo "Running GPT on $dataset version $version..."
    srun python baselines/gpt/train_gpt.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    srun python baselines/gpt/sample_gpt.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    srun python evaluation/run_evaluation.py --output_path $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    find "$LOG_DIR/utility_models" -type f -name "*.pt" -delete
    echo "Deleted .pt files in $LOG_DIR/utility_models"
}

run_eva() {
    LOG_DIR="$BASE_LOG_DIR/${dataset}_${dataset_version}_eva"
    mkdir -p "$LOG_DIR"
    echo "Running EVA on $dataset version $version..."
    srun python baselines/eva/train_eva.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    srun python baselines/eva/sample_eva.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    srun python evaluation/run_evaluation.py --output_path $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    find "$LOG_DIR/utility_models" -type f -name "*.pt" -delete
    echo "Deleted .pt files in $LOG_DIR/utility_models"
}

run_synteg() {
    LOG_DIR="$BASE_LOG_DIR/${dataset}_${dataset_version}_synteg"
    mkdir -p "$LOG_DIR"
    echo "Running SynTEG on $dataset version $version..."
    srun python baselines/synteg/dependency_learning.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version

    srun python baselines/synteg/export_condition.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version

    srun python baselines/synteg/condition_simulation.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version

    srun python baselines/synteg/generate_data.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version

    srun python evaluation/run_evaluation.py --output_path $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    find "$LOG_DIR/utility_models" -type f -name "*.pt" -delete
    echo "Deleted .pt files in $LOG_DIR/utility_models"
}

run_haloCoarse() {
    LOG_DIR="$BASE_LOG_DIR/${dataset}_${dataset_version}_haloCoarse"
    mkdir -p "$LOG_DIR"
    echo "Running HALOCoarse on $dataset version $version..."
    srun python baselines/haloCoarse/train_haloCoarse.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    srun python baselines/haloCoarse/sample_haloCoarse.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    srun python evaluation/run_evaluation.py --output_path $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version 
    find "$LOG_DIR/utility_models" -type f -name "*.pt" -delete
    echo "Deleted .pt files in $LOG_DIR/utility_models"
}

run_halo() {
    LOG_DIR="$BASE_LOG_DIR/${dataset}_${dataset_version}_halo"
    mkdir -p "$LOG_DIR"
    echo "Running HALO on $dataset version $version..."
    srun python baselines/halo/train_halo.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    srun python baselines/halo/sample_halo.py --LOG_DIR $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version
    srun python evaluation/run_evaluation.py --output_path $LOG_DIR \
        --dataset $dataset \
        --dataset_version $dataset_version \
    find "$LOG_DIR/utility_models" -type f -name "*.pt" -delete
    echo "Deleted .pt files in $LOG_DIR/utility_models"
}

# Execute baselines
if [ "$RUN_LSTM" = true ]; then
    run_lstm
fi

if [ "$RUN_GPT" = true ]; then
    run_gpt
fi

if [ "$RUN_EVA" = true ]; then
    run_eva
fi

if [ "$RUN_SynTEG" = true ]; then
    run_synteg
fi

if [ "$RUN_HALOCOARSE" = true ]; then
    run_haloCoarse
fi

if [ "$RUN_HALO" = true ]; then
    run_halo
fi
# Finalize
echo "Job finished at: $(date '+%Y-%m-%d-%H_%M_%S')"
mv "output/slurm-${SLURM_JOB_ID}.output" "$BASE_LOG_DIR/slurm-${SLURM_JOB_ID}.output"
mv "output/slurm-${SLURM_JOB_ID}.error" "$BASE_LOG_DIR/slurm-${SLURM_JOB_ID}.error"
