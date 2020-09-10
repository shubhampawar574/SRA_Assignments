from PIL import Image
import numpy as np

def convolve(image, kernel):
    data = np.array(image)
    output = np.zeros_like(data)
    image_padded = np.zeros((data.shape[0] + 4, data.shape[1] + 4, data.shape[2]))
    image_padded[2:-2, 2:-2, :] = data

    # Loop over every pixel of the image
    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            for z in range(data.shape[2]):
                # element-wise multiplication of the kernel and the image
                output[y, x, z] = (kernel * image_padded[y: y+5, x: x+5, z]).sum()            
    return output

input_image = Image.open('filter.png')

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

kernel = gaussian_kernel(5, sigma=1)
imagegaussianblur = convolve(input_image, kernel=kernel)

gausssian_blur_image = Image.fromarray(imagegaussianblur.astype("uint8"))
gausssian_blur_image.show()
gausssian_blur_image.save('gausssian_blur_output.png')

imageboxfilter = convolve(input_image, kernel=np.array([[1/27, 1/27, 1/27, 1/27, 1/27], 
                                                          [1/27, 1/27, 1/27, 1/27, 1/27], 
                                                          [1/27, 1/27, 1/27, 1/27, 1/27], 
                                                          [1/27, 1/27, 1/27, 1/27, 1/27], 
                                                          [1/27, 1/27, 1/27, 1/27, 1/27]]))
box_filter_image = Image.fromarray(imageboxfilter.astype("uint8"))
box_filter_image.show()
box_filter_image.save('box_blur_output.png')

imagesharpen = convolve(input_image, kernel=np.array([[1/25,1/25,1/25,1/25,1/25], [1/25,1/25,1/25,1/25,1/25], [1/25,1/25,-1,1/25,1/25], 
                                                            [1/25,1/25,1/25,1/25,1/25], [1/25,1/25,1/25,1/25,1/25]]))
sharpen_image = Image.fromarray(imagesharpen.astype("uint8"))
sharpen_image.show()
sharpen_image.save('sharpen_output.png')
