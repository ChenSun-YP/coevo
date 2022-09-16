
# ResNet56 on CIFAR-10
python main.py --dataset cifar10 --arch resnet56  \
                  --ft_epoch 100 --lr_milestone 50 \
                  --dict_path ./models/resnet56.th  --pop_init_rate 0.95  \
                  --prune_limitation 0.90 --batch-size 128 --valid_ratio 0.8  \
                  --run_epoch 20
wait
echo "Done"


# VGG16 on CIFAR-10
python main.py --dataset cifar10 --arch vgg  \
                  --ft_epoch 100 --lr_milestone 50 \
                  --dict_path ./models/vgg16.th  --pop_init_rate 0.95  \
                  --prune_limitation 0.90 --batch-size 128 --valid_ratio 0.8  \
                  --run_epoch 20
wait
echo "Done"


# ResNet50 on ImageNet (ImageNet is a very large dataset, you can get it from https://www.image-net.org/)
python main.py --dataset ImageNet --arch resnet50  \
                  --ft_epoch 60 --lr_milestone 20 40 50 \
                  --data ./data/ImageNet  --pop_init_rate 0.9  \
                  --prune_limitation 0.85 --batch-size 256 --valid_ratio 0.99  \
                  --run_epoch 20
wait
echo "Done"


# ResNet34 on ImageNet (ImageNet is a very large dataset, you can get it from https://www.image-net.org/)
python main.py --dataset ImageNet --arch resnet34  \
                  --ft_epoch 60 --lr_milestone 20 40 50 \
                  --data ./data/ImageNet  --pop_init_rate 0.9  \
                  --prune_limitation 0.85 --batch-size 256 --valid_ratio 0.99  \
                  --run_epoch 20
wait
echo "Done"

