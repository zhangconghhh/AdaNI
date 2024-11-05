## AdaNI: Adaptive Noise Injection to improve adversarial robustness



This repository contains an implementation corresponding to our CVIU 2024 paper: "[AdaNI: Adaptive Noise Injection to improve adversarial robustness](https://www.sciencedirect.com/science/article/abs/pii/S1077314223002357)". 


This repository includes PyTorch implementation of:

- Adversarial attacks 
    - FGSM
    - PGD
    - EOT (Expectation Over Transformations)
- Baseline models used in experiments
- AdaNI Modules




## Quick Start


To train the AdaNI model, run:
```
sh adani.sh
```
To evaluate the model after training, run:

```
python evaluation.py
```


## Cite the paper
If you find our work useful, please cite it as follows:
```bibtex
@article{LI2024103855,
title = {AdaNI: Adaptive Noise Injection to improve adversarial robustness},
author = {Yuezun Li and Cong Zhang and Honggang Qi and Siwei Lyu},
journal = {Computer Vision and Image Understanding},
volume = {238},
pages = {103855},
year = {2024},
issn = {1077-3142},
}
```
## Acknowledgements

The codes are modified from [Learn2Perturb](https://github.com/ArmenJeddi/Learn2Perturb). Thanks for their open source.
