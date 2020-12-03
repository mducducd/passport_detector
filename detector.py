import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import subprocess

# define helper functions
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

if __name__ == "__main__":
    program='./darknet'
    arguments=('detector test cfg/obj.data.txt cfg/yolov4-obj.cfg models/yolov4-obj_6000.weights -ext_output < image.txt > result.txt -thresh 0.5')
    subprocess.call([program, arguments])

    plt.imshow('predictions.jpg')
    plt.show()