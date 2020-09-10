from PIL import Image
import numpy as np

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def convolve(data, kernel):
    output = np.zeros_like(data)
    image_padded = np.zeros((data.shape[0] + kernel.shape[0] - 1, data.shape[1] + kernel.shape[1] - 1))
    image_padded[int((kernel.shape[0] - 1)/2):-int((kernel.shape[0] - 1)/2),
                   int((kernel.shape[0] - 1)/2):-int((kernel.shape[0] - 1)/2)] = data

    # Loop over every pixel of the image
    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y: y+kernel.shape[0], x: x+kernel.shape[1]]).sum()
                   
    return output

def non_max_suppression(img,D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def threshold(img, weak,strong,lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(weak)
    strong = np.int32(strong)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res

def hysteresis(img, weak, strong):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

input_image = Image.open('edge-detection2.jpg')
data = np.array(input_image)
gray_image = rgb2gray(data)
kernel = gaussian_kernel(5, sigma=1)
imagegaussianblur = convolve(gray_image, kernel=kernel)

# applying sobel filters to get image gradient
intensity_x=convolve(imagegaussianblur,kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32))  #intensity along x axis
intensity_y=convolve(imagegaussianblur,kernel=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],np.float32))  #intensity along y axis
image=np.hypot(intensity_x,intensity_y)                     #finding the sum of intensities
image= image / image.max() * 255                                  
theta=np.arctan2(intensity_y,intensity_x)

# non maximum suppression
image=non_max_suppression(image,theta)

# applying double threshold
weak=75
strong=255
image=threshold(image,weak,strong,0.05,0.15)
# applying hysteresis
image=hysteresis(image,weak,strong)


image=Image.fromarray(image)
image.show()