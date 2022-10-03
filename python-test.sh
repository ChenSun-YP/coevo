#!/bin/bash
#SBATCH --gres=gpu:4       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=12G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=mrsunchen0110@gmail.com
#SBATCH --mail-type=ALL

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip --version
pip install --no-index torch==1.9.1 torchvision==0.9.1 
pip install thop

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"


nvidia-smi                        # you can use 'nvidia-smi' for a test
# ResNet56 on CIFAR-10
python main.py --dataset cifar10 --arch resnet56  \
                  --ft_epoch 20 --lr_milestone 50 \
                  --dict_path ./models/resnet56.th  --pop_init_rate 0.95  \
                  --prune_limitation 0.80 --batch-size 128 --valid_ratio 0.8  \
                  --run_epoch 2  --use_crossover
wait
echo "Done"

