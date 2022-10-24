
This repo apply the coevo pruning method mentioned in the paper *Wu, Jia-Liang, et al. "Robust Neural Network Pruning by Cooperative Coevolution." International Conference on Parallel Problem Solving from Nature. Springer, Cham, 2022.* onto Deep Spiking Neural networks.





![prune plot 1 ](https://github.com/ChenSun-YP/coevo/blob/main/plot.jpg.png)




```<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate ```
python main_snn_2.py -T 4 -device cuda:0 -b 128 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8 -b 20
```
cd /Applications/Python\ 3.9/
./Install\ Certificates.command
```


### Requirements

- python==3.8.3
- torch==1.8.2
- torchvision==0.9.2 
- thop==0.0.31
- prefetch_generator==1.0.1


### Usage
#### Pruning on CIFAR-10

```shell
# ResNet56
python pruning.py --dataset cifar10 --arch resnet56  \
                  --ft_epoch 100 --lr_milestone 50 \
                  --dict_path ./models/resnet56.th  --pop_init_rate 0.95  \
                  --prune_limitation 0.90 --batch-size 128 --valid_ratio 0.8  \
                  --run_epoch 20
# VGG16
python pruning.py --dataset cifar10 --arch vgg  \
                  --ft_epoch 100 --lr_milestone 50 \
                  --dict_path ./models/vgg16.th  --pop_init_rate 0.95  \
                  --prune_limitation 0.90 --batch-size 128 --valid_ratio 0.8  \
                  --run_epoch 20
```

#### Pruning On ImageNet

```shell
# ResNet-50
python pruning.py --dataset ImageNet --arch resnet50  \
                  --ft_epoch 60 --lr_milestone 20 40 50 \
                  --data ./data/ImageNet  --pop_init_rate 0.9  \
                  --prune_limitation 0.85 --batch-size 256 --valid_ratio 0.99  \
                  --run_epoch 20
# ResNet-34
python pruning.py --dataset ImageNet --arch resnet34  \
                  --ft_epoch 60 --lr_milestone 20 40 50 \
                  --data ./data/ImageNet  --pop_init_rate 0.9  \
                  --prune_limitation 0.85 --batch-size 256 --valid_ratio 0.99  \
                  --run_epoch 20
```

You can find all the commands in the file 'run.sh' .
