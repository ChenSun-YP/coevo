#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch torchvision torchtext torchaudio
pip install thop
pip install prefetch_generator
nvidia-smi                        # you can use 'nvidia-smi' for a test
python pytorch-test.py
# ResNet56 on CIFAR-10
python main.py --dataset cifar10 --arch resnet56  \
                  --ft_epoch 100 --lr_milestone 50 \
                  --dict_path ./models/resnet56.th  --pop_init_rate 0.95  \
                  --prune_limitation 0.90 --batch-size 128 --valid_ratio 0.8  \
                  --run_epoch 2
wait
echo "Done"

