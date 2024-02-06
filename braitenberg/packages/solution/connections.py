from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    for i in range (0, 320):
        res[240:, i] = (i)/3200

    for i in range (320, 640):
        res[240:, i] = -(640-i)/3200
    # res[100:150, 100:150] = 1
    # res[300:, 200:] = 1
    # ---
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    for i in range (0, 320):
        res[240:, i] = -i/3200

    for i in range (320, 640):
        res[240:, i] = (640-i)/3200
    #res[100:150, 100:300] = -1
    #res[:, :] = -0.01
    # ---
    return res
