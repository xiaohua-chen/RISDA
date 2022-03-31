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


## CIFAR
In the code, we calculate the accuracy, which is different from that in the paper.
```
CIFAR-LT-100,long-tailed imabalance ratio of 200
python RISDA.py --gpu 3 --lr 0.1 --alpha 0.5 --beta 1 --imb_factor 0.005 --dataset cifar100 --num_classes 100 --save_name simple --idx cifar_im200
```
```
CIFAR-LT-100,long-tailed imabalance ratio of 100
python RISDA.py --gpu 3 --lr 0.1 --alpha 0.5 --beta 0.75 --imb_factor 0.01 --dataset cifar100 --num_classes 100 --save_name simple --idx cifar_im100
```
More details will be uploaded soon.


## Acknowledgements
Some codes in this project are adapted from [MetaSAug](https://github.com/BIT-DA/MetaSAug) and [ISDA](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks). We thank them for their excellent projects.
   
    
## Citation

If you find this code useful for your research, please cite our paper.

>@article{chen2021imagine,
  title={Imagine by Reasoning: A Reasoning-Based Implicit Semantic Data Augmentation for Long-Tailed Classification},
  author={Chen, Xiaohua and Zhou, Yucan and Wu, Dayan and Zhang, Wanqian and Zhou, Yu and Li, Bo and Wang, Weiping},
  journal={arXiv preprint arXiv:2112.07928},
  year={2021}
}
