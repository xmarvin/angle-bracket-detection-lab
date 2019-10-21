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

def get_boundaries(imgray):
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    _, contours, _  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cords in contours:
         x = [a[0] for a in np.squeeze(cords)]
         y = [a[1] for a in np.squeeze(cords)]
         minx, maxx = np.min(x), np.max(x)
         miny, maxy = np.min(y), np.max(y)
         boxes.append((minx,miny,maxx,maxy))
    return boxes

def disable_bracket(image, mask, brackets):
    brackets_len = len(brackets)
    n = np.random.randint(brackets_len)
    selected_bracket_ids = np.random.choice(range(brackets_len), n, replace=False)
    selected_brackets = brackets[selected_bracket_ids]
    img_disabled = image.copy()
    mask_disabled = mask.copy()
    for (x1, y1, x2, y2) in selected_brackets:
        img_disabled = cv2.rectangle(img_disabled, (x1,y1), (x2,y2), (255, 255,255), -1)
        mask_disabled = cv2.rectangle(mask_disabled, (x1,y1), (x2,y2), (0, 0,0), -1)
    return img_disabled, mask_disabled

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