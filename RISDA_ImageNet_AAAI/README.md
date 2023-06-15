# Image Classification on ImageNet

## Requirements
- python 3.7
- pytorch 1.0.1
- torchvision 0.2.2


## Run

Train ResNet-50 on ImageNet-LT

```
CUDA_VISIBLE_DEVICES=1,0 python imagenet_ISDA_train.py  /datapath/ILSVRC2012_LT/ --model resnet50 --batch-size 128 --lr 0.1 --epochs 100 --alpha_0 0 --beta_0 7.5 --workers 1  --world-size 1 --rank 0  --stage1 80 --stage2 90 

```

Test ResNet-50 on ImageNet-LT

```
CUDA_VISIBLE_DEVICES=1,0 python imagenet_ISDA_train.py  /datapath/ILSVRC2012_LT/ --model resnet50 --batch-size 128 --lr 0.1 --epochs 100 --alpha_0 7.5 --beta_0 0 --workers 1  --world-size 1 --rank 0  --stage1 80 --stage2 90 --evaluate checkpoint/best.pth.tar

```










