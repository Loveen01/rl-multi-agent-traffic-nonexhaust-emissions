# The first Multi-Agent RL traffic optimisation with pedestrians (to reduce non-exhaust emmissions)


1. Create RL environment
The simulation infrastructure we use is SUMO. However, in order to create an envioronment with SUMO, we will use its API to construct its environment, namely SUMO-RL. 

SUMO-RL: implements PettingZoo AECEnv environment interface, providing interface to work with SUMO 

PettingZoo contains the AECEnv environment interface, for which projects like SUMO and others implement.


RLlib is an industry-grade open-source reinforcement learning library. It is a part of Ray, a popular library for distributed ML and scaling python applications. [https://pettingzoo.farama.org/tutorials/rllib/]


## Requirements
* Python3==
* [Tensorflow](http://www.tensorflow.org/install)==
* [SUMO](http://sumo.dlr.de/wiki/Installing)>=


## Citation
If you find this useful in your research, please cite our paper "Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control" ([early access version](https://ieeexplore.ieee.org/document/8667868), [preprint version](https://arxiv.org/pdf/1903.04527.pdf)):
~~~
@article{chu2019multi,
  title={Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control for reduced EEE},
  author={Loveen Omar{\`a}},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}
~~~