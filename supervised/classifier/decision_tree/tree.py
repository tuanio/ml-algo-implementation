import numpy as np
from dataclasses import dataclass
from collections import deque # using deque for storing tree

@dataclass
class TreeNode:
    depth: int
    entropy: float
    data: np.ndarray

@dataclass
class DecisionTree:
    