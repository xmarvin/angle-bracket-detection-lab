from model.u_net import get_unet_256, get_unet_512, get_unet_128
import cv2
import numpy as np
input_size = 512

max_epochs = 50
batch_size = 32

threshold = 0.5

if input_size == 256:
  model_factory = get_unet_256
elif input_size == 128:
  model_factory = get_unet_128
else:
  model_factory = get_unet_512

def simplify_image(image):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    #gray = cv2.GaussianBlur(image, (3, 3), 0)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rectKernel)
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    return gradX