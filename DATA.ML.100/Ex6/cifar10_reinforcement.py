import gym
import pickle
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()
env.render()
