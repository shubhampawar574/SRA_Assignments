# SRA_Assignments
# Assignments Description
We take input as an image and perform operations like image rotation, blurring and sharpening vertical, horizontal, sobel and canny edge detection, morphological transformations,
masking, region of interest

# Tools:
Numpy and pillow libraries in python are used
# 1)Image rotation:
Here, I have used the trigonometric formula to find new co-ordinates when axes are rotated by any angle about the origin:
X = xcos(theta)+ysin(theta) and 
Y = -xsin(theta)+ycos(theta)
. Before applying this formula we set centre of image as origin. The code is written such that every pixel in image is rotated by angle theta.
The code, input and output images are in above rotation folder.
# 2)Blurring and sharpening:
a) For gaussian blur, I have used a function that generates a gaussian kernel from net and applied it on image to get gaussian blur image. 
b) For box blur, I have used a 5*5 kernel available om net and applied it on image to get box blur image.
c) For sharpening, I have used a 5*5 kernel available om net and applied it on image to get sharpened image.
The code, input and output images are in above kernels folder.
# 3)Edge detection:
a) For horizontal and vertical edge detection, we first convert image to gray image and apply 3*3 kernels to get horizontal and vertical edge detection output. 
b) For sobel edge detection,  we first convert image to gray image and apply 3*3 kernels in X and Y direction individually. Next, we combine them with hypot function in numpy to 
get sobel edge detection output.
c) For canny edge detection, https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123 this link helped me understand canny edge detection.
The code, input and output images are in above edge_detection folder.
# 4)Morphological Transformations:
a)Erosion and dilation:
First convert input image to ray image. then, we convert it to binary image wuth bitwise &. We use structuring element (i.e. 3*3 matrix). 
https://www.ques10.com/p/29867/explain-the-dilation-and-erosion-with-example/ . This link helped understand better. 
The code, input and output images are in above morphological transformations folder.
# 5)Mask:
Here, we first convert image from BGR tor RGB and to HSV. Then we find pixels within range of colour to be detected ,perform bitwise_and operation in numpy and convert it back to RGB.
The code, input and output images are in above mask folder.
# 6) ROI:
Here with trial and error we detect object pixels range and paste it anywhere on the image we want. 
The code, input and output images are in above roi folder.




