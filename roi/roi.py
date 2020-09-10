from PIL import Image
import numpy as np

input_image = Image.open('roi.jpg')
data = np.array(input_image)

mask = np.zeros((150, 154, data.shape[2]), dtype = np.uint8)
mask[:, :, :] = data[890:1040, 1035:1189,:]
data[865:1015, 325:479, :] = mask[:, :, :]

image = Image.fromarray(data)
image.show()
