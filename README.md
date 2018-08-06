# Privacy-AdversarialLearning
## TensorFlow Code for 'Towards Privacy-Preserving Visual Recognition via Adversarial Training: A Pilot Study'

## Introduction

TensorFlow Implementation of our ECCV 2018 paper ["Towards Privacy-Preserving Visual Recognition via Adversarial Training: A Pilot Study"](https://arxiv.org/abs/1807.08379).

This paper aims to improve privacy-preserving visual recognition, an increasingly demanded feature in smart camera applications, by formulating a unique adversarial training framework. 

The proposed framework explicitly learns a degradation transform for the original video inputs, in order to optimize the trade-off between target task performance and the associated privacy budgets on the degraded video. A notable challenge is that the privacy budget, often defined and measured in task-driven contexts, cannot be reliably indicated using any single model performance, because a strong protection of privacy has to sustain against any possible model that tries to hack privacy information. 

Such an uncommon situation has motivated us to propose two strategies to enhance the generalization of the learned degradation on protecting privacy against unseen hacker models. Novel training strategies, evaluation protocols, and result visualization methods have been designed accordingly. 

Two experiments on privacy-preserving action recognition, with privacy budgets defined in various ways, manifest the compelling effectiveness of the proposed framework in simultaneously maintaining high target task (action recognition) performance while suppressing the privacy breach risk.

## To do items

### Privacy Preserving in Smart Home

- [ ] SBU
- [ ] UCF-101 / VISPR

### Privacy Preserving in Data Sharing

- [ ] AFLW

## Dependencies

Python 3.5
* [TensorFlow 1.8.0](https://www.tensorflow.org/)

## Citation

If you find this code useful, please cite the following paper:

    @article{privacy_adversarial_2018, 
      title={Towards Privacy-Preserving Visual Recognition via Adversarial Training: A Pilot Study}, 
      journal={ECCV}, 
      author={Wu, Zhenyu and Wang, Zhangyang and Wang, Zhaowen and Jin, Hailin}, 
      year={2018}
    }
