from PIL import Image, ImageOps
import numpy as np

def erosion(image, kernel):
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0]+kernel.shape[0]-1,image.shape[1] + kernel.shape[1]-1))
    image_padded[kernel.shape[0]-2:-1:,kernel.shape[1]-2:-1:] = image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            sum=(kernel * image_padded[y: y+kernel.shape[0], x: x+kernel.shape[1]]).sum()
            if sum==5:
                output[y,x]=1
            else :
                output[y,x]=0
    return output
            
input_image = Image.open('morphological.png')
data = np.array(input_image)

r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
binary_image = (gray>=128) & (gray<255)

image=erosion(binary_image,kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
image=Image.fromarray(image)
image.show()