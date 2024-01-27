import numpy as np
import cv2


def np_imread(img_file):
    cv_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img