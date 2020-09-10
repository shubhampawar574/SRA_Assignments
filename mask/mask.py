import cv2
import numpy as np
from PIL import Image

input_image = Image.open('mask.jpg')
im = np.array(input_image)
lower = np.array([80,80,0])
upper = np.array([179,255,255])
bgr_image=np.zeros_like(im)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        bgr_image[i][j][0]=im[i][j][2]
        bgr_image[i][j][1]=im[i][j][1]
        bgr_image[i][j][2]=im[i][j][0]

hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

mask=np.zeros((hsv.shape[0],hsv.shape[1],hsv.shape[2]),dtype=np.uint8)
for i in range(hsv.shape[0]):
    for j in range(hsv.shape[1]):
        if lower[0]<=hsv[i][j][0]<=upper[0] and lower[1]<=hsv[i][j][1]<=upper[1] and lower[2]<=hsv[i][j][2]<=upper[2]:
            mask[i][j][0] = hsv[i][j][0]
            mask[i][j][1] = hsv[i][j][1]
            mask[i][j][2] = hsv[i][j][2]
image=np.bitwise_and(mask, hsv)
image= cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
image= Image.fromarray(image.astype("uint8"))
image.show()
    


