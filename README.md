# Imagine by Reasoning: A Reasoning-Based Implicit Semantic Data Augmentation for Long-Tailed Classification (AAAI 2022)

## Prerequisite
* PyTorch >= 1.2.0
* Python3
* torchvision
* argparse
* numpy

## Dataset

* Imbalanced CIFAR. The original data will be downloaded and converted by imbalancec_cifar.py
* Imbalanced ImageNet
* The paper also reports results on iNaturalist 2018(https://github.com/visipedia/inat_comp). 


## CIFAR100
In the code, we calculate the accuracy, which is different from that in the paper.
```
CIFAR-LT-100,long-tailed imabalance ratio of 200
python RISDA.py --gpu 3 --lr 0.1 --alpha 0.5 --beta 1 --imb_factor 0.005 --dataset cifar100 --num_classes 100 --save_name simple --idx cifar_im200
```
```
CIFAR-LT-100,long-tailed imabalance ratio of 100
python RISDA.py --gpu 3 --lr 0.1 --alpha 0.5 --beta 0.75 --imb_factor 0.01 --dataset cifar100 --num_classes 100 --save_name simple --idx cifar_im100
```
# Image Classification on ImageNet

## Run

Train ResNet-50 on ImageNet-LT

```
CUDA_VISIBLE_DEVICES=1,0 python imagenet_ISDA_train.py  /datapath/ILSVRC2012_LT/ --model resnet50 --batch-size 128 --lr 0.1 --epochs 100 --alpha_0 0 --beta_0 7.5 --workers 1  --world-size 1 --rank 0  --stage1 80 --stage2 90 

```

Test ResNet-50 on ImageNet-LT

```
CUDA_VISIBLE_DEVICES=1,0 python imagenet_ISDA_train.py  /datapath/ILSVRC2012_LT/ --model resnet50 --batch-size 128 --lr 0.1 --epochs 100 --alpha_0 0 --beta_0 7.5 --workers 1  --world-size 1 --rank 0  --stage1 80 --stage2 90 --evaluate checkpoint/best.pth.tar

```


More details will be uploaded soon.


## Acknowledgements
Some codes in this project are adapted from [MetaSAug](https://github.com/BIT-DA/MetaSAug) and [ISDA](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks). We thank them for their excellent projects.
   
    
## Citation

If you find this code useful for your research, please cite our paper.
```
@inproceedings{chen2021imagine,
  title={Imagine by Reasoning: A Reasoning-Based Implicit Semantic Data Augmentation for Long-Tailed Classification},
  author={Chen, Xiaohua and Zhou, Yucan and Wu, Dayan and Zhang, Wanqian and Zhou, Yu and Li, Bo and Wang, Weiping},
  booktitle = {Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022}
}
```
