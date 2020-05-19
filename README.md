# Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition from a Domain Adaptation Perspective
[[Arxiv](https://arxiv.org/abs/2003.10780)]

This is PyTorch implementation of the above CVPR 2020 paper.

#### Abstract

Object frequency in the real world often follows a power law, leading to a mismatch between datasets with long-tailed class distributions seen by a machine learning model and our expectation of the model to perform well on all classes. We analyze this mismatch from a domain adaptation point of view. First of all, we connect existing class-balanced methods for long-tailed classification to target shift, a well-studied scenario in domain adaptation. The connection reveals that these methods implicitly assume that the training data and test data share the same class-conditioned distribution, which does not hold in general and especially for the tail classes. While a head class could contain abundant and diverse training examples that well represent the expected data at inference time, the tail classes are often short of representative training data. To this end, we propose to augment the classic class-balanced learning by explicitly estimating the differences between the class-conditioned distributions with a meta-learning approach.

# Dependency

PyTorch0.4

# Dataset

imbalanced CIFAR 10 and 100

# Training

To train CIFAR-LT dataset, go C-LT/ folder and run

e.g. to train CIFAR10-LT with an imabalance factor of 200, run

```
python main.py --dataset cifar10 --num_classes 10 --imb_factor 0.005
```
