from typing import Tuple

import numpy as np
import time

# start_time = datetime.now()


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    
    # for i in range (0, 320):
    #     res[240:, i] = (i)/3200

    # for i in range (320, 640):
    #     res[240:, i] = -(640-i)/3200

    # res[200:, 310:430] = -1
    # res[150:, 430:500] = -0.6
    # res[250:, 500:575] = -0.4
    # res[350:, 575:640] = -0.2

    # res[200:, 230:310] = 1
    # res[150:, 150:230] = 0.6
    # res[250:, 75:150] = 0.4
    # res[300:, 0:75] = 0.2
    # res[200:, 320:370] = -0.5

    # res[200:, 310:540] = -1
    # res[300:, 540:] = -1

    # end_time = datetime.now()
    # time_delay = (end_time - start_time)
    # if time_delay.total_seconds() <= 0.3:
    #     res[200:, 295:345] = 1
    # else:
    res[200:, 150:310] = 1
    res[300:, :150] = 1

    

    # res[100:150, 100:150] = 1
    # res[300:, 200:] = 1
    # ---
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values

    # for i in range (0, 320):
    #     res[240:, i] = -i/3200

    # for i in range (320, 640):
    #     res[240:, i] = (640-i)/3200

    # res[200:, 310:430] = 1
    # res[150:, 430:500] = 0.6
    # res[250:, 500:575] = 0.4
    # res[350:, 575:640] = 0.2

    # res[200:, 230:310] = -1
    # res[150:, 150:230] = -0.6
    # res[250:, 75:150] = -0.4
    # res[300:, 0:75] = -0.2

    # end_time = datetime.now()
    # time_delay = (end_time - start_time)
    # if time_delay.total_seconds() <= 0.3:
    #     res[200:, 295:345] = -1
    # else:
    res[200:, 310:490] = 1
    res[300:, 490:] = 1

    # res[200:, 100:310] = -1
    # res[300:, :100] = -1
    # res[200:, 270:320] = -0.5
    

    #res[100:150, 100:300] = -1
    #res[:, :] = -0.01
    # ---
    return res
