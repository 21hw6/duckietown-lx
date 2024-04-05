from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    steer_matrix_left = np.zeros(shape=shape, dtype="float32")
    steer_matrix_left[200:, 150:310] = 1
    steer_matrix_left[300:, :150] = 1
    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    steer_matrix_right = np.zeros(shape=shape, dtype="float32")
    steer_matrix_right[200:, 310:490] = 1
    steer_matrix_right[300:, 490:] = 1
    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    H = np.array([-4.137917960301845e-05, -0.00011445854191468058, -0.1595567007347241, 
              0.0008382870319844166, -4.141689222457687e-05, -0.2518201638170328, 
              -0.00023561657746150284, -0.005370140574116084, 0.9999999999999999])

    H = np.reshape(H,(3, 3))
    Hinv = np.linalg.inv(H)

    points = Hinv @ np.array([400, 0, 1])

    horizon = int(points[1]/points[2])

    mask_ground = np.ones(img.shape, dtype=np.uint8) # TODO: CHANGE ME

    # Mask out everything above our horizon line
    mask_ground[:horizon, :] = 0

    # TODO: Identify a setting for the standard deviation that removes noise while not eliminating too much valid content.
    sigma = 2 # CHANGE ME

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    # Convolve the image with the Sobel operator (filter) to compute the numerical derivatives in the x and y directions
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    threshold = 75 # CHANGE ME

    mask_mag = (Gmag > threshold)

    white_lower_hsv = np.array([0, 0, 150])
    white_upper_hsv = np.array([190, 50, 255])
    yellow_lower_hsv = np.array([0, 100, 100]) #[0, 100, 100]
    yellow_upper_hsv = np.array([50, 255, 255]) #[50, 255, 255]

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    width = img.shape[1]
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    # TODO: implement your own solution here
    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
