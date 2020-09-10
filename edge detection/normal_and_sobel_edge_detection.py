from PIL import Image
import numpy as np

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def convolve(data, kernel):
    output = np.zeros_like(data)
    image_padded = np.zeros((data.shape[0] + 2, data.shape[1] + 2))
    image_padded[1:-1, 1:-1] = data

    # Loop over every pixel of the image
    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y: y+3, x: x+3]).sum()         
    return output

input_image = Image.open('edge-detection2.jpg')
data = np.array(input_image)
gray_image = rgb2gray(data)

#horizontal edge detection
image_edge1 = convolve(gray_image, kernel=np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
image_edge1 = Image.fromarray(image_edge1.astype("uint8"))
image_edge1.show()
image_edge1.save('horizontal_edge_detection_output.jpg')

#vertical edge detection
image_edge2 = convolve(gray_image, kernel=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
image_edge2 = Image.fromarray(image_edge2.astype("uint8"))
image_edge2.show()
image_edge1.save('vertical_edge_detection_output.jpg')


#kernel to be used for sobel vertical edge detection
image_sobel_edge1 = convolve(gray_image, kernel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))

#kernel to be used for sobel vertical edge detection
image_sobel_edge2 = convolve(gray_image, kernel=np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

image=np.hypot(image_sobel_edge1,image_sobel_edge2)
image=Image.fromarray(image)
image.show()
image.save('sobel_output.jpg')
