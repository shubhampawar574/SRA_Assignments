from PIL import Image
import numpy as np
import math

im = Image.open('rotate.png')
degree_angle = int(input("Enter angle by which image has to be rotated in counter clockwise direction: "))
radians_angle = degree_angle*math.pi/180
data = np.array(im)
h, w, c = data.shape
if h>w:
	size=h
else:
	size=w
final_im = np.zeros((size, size, c))
fh, fw, fc = final_im.shape
cy = fh/2
cx = fw/2
min_x = 0
min_y = 0
for i in range(fh):
	for j in range(fw):
		x= round(cx -(i-cx)*math.sin(radians_angle) + (j-cy)*math.cos(radians_angle))
		y= round(cy + (i-cx)*math.cos(radians_angle) + (j-cy)*math.sin(radians_angle))
		if (0<=x<w and 0<=y<h):
			final_im[i][j][:] = data[y][x][:]
		
image = Image.fromarray(final_im.astype("uint8"))
image.show()
image.save('rotation_output.png')
