import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

e = np.eye(3)
a = np.array([[2, 2, 2]])
b = np.concatenate((e, a.T), axis=1)
print(b.shape)
c=(np.ones((3,1)))
print(b.shape)
print(c.shape)
f = np.concatenate((b, c), axis=1)
print(f)
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6]])
# c = np.concatenate((a, b.T), axis=1)
# print(c)