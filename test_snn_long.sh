#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8G  # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-13:00
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=mrsunchen0110@gmail.com
#SBATCH --mail-type=ALL

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 --version
pip3 install --no-index torch torchvision==0.12.0 torchtext torchaudio

pip3 install thop
pip3 install --no-index torchsummary
pip3 install --no-index  tensorboard
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install
cd ..
nvidia-smi                        # you can use 'nvidia-smi' for a test
# ResNet56 on CIFAR-10
python main_snn_2.py -T 4 -device cuda:0 -b 128 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8 -b 20 -out-dir ./logs/testsnn/

# python  main_snn.py --dataset ImageNet --arch vgg  \
#                  --ft_epoch 100 --lr_milestone 50   --data ./data/ImageNet  \
#                  --dict_path ./models/vgg16.th  --pop_init_rate 0.95  \
#                  --prune_limitation 0.90 --batch-size 128 --valid_ratio 0.8  \
#                  --run_epoch 2
#
#python main_snn.py --dataset cifar10 --arch vgg  \
#                  --ft_epoch 100 --lr_milestone 50 \
#                  --dict_path ./models/vgg16.th  --pop_init_rate 0.95  \
#                  --prune_limitation 0.90 --batch-size 128 --valid_ratio 0.8  \
#                  --run_epoch 20

wait
echo "Done"


