# Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition from a Domain Adaptation Perspective

This is PyTorch implementation of the above CVPR 2020 paper.

# Dependency

PyTorch0.4

# Dataset

imbalanced CIFAR 10 and 100

# Training

To train CIFAR-LT dataset, go C-LT/ folder and run

e.g. to train CIFAR10-LT with an imabalance factor of 200, run

'''
python main.py --dataset cifar10 --num_classes 10 --imb_factor 0.005
'''

# Reference
