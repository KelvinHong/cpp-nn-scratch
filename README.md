# C++ implementation of Neural Network Capabilities from Scratch

This project implements neural network nodes structure with forward and backward passes, 
it automatically creates Directed Acyclic Graph (DAG) with referencing to PyTorch Autograd 
structure. Moreover, gradient can be back-propagated layer by layer to the leaf nodes, 
by handcrafting custom gradient calculation methods. We uses C++ Eigen Library for 
common matrix utilities (for now), considering we only uses batched 1D inputs. 

This is my hobby project so please don't use this for any kind of production level 
tasks. It only serves as a purpose of strengthen my deep learning understanding 
and C++ skills.  

## Requirements

Install C++ Eigen Library.

Install C++ Boost Library (for enum to string capabilities).

## Demonstration

On Windows or Linux, run make:
```
make
```

to compile the executables. 

Run `./autotest.exe` to verify all functionalities are good.

Run `./firstModel.exe` to run model training on the Wine Quality Dataset (WIP).

## TODO List

Implement model saving and loading

Implement visualization of backward graph.

Implement Batchnorm.