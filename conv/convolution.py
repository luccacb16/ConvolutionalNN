import numpy as np
from PIL import Image
import math

# MaxPooling
import pooling as pl

# Gaussian blur 3x3 kernel
blur = np.array([[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]]) / 16

# Edge detection kernel
edge = np.array([[-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]])

# Sharpen kernel
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

# Identity kernel
identity = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

laplace_gauss = np.array([[0, 0, -1, 0, 0],
                          [0, -1, -2, -1, 0],
                          [-1, -2, 16, -2, -1],
                          [0, -1, -2, -1, 0],
                          [0, 0, -1, 0, 0]])

def convolve(img: np.array, kernel: np.array, stride: int = 1, padding: int = 1) -> np.array:

    # Rotaciona o kernel em 180 graus
    kernel = np.rot90(kernel, 2)

    # Padding
    img_padded = np.pad(
        img, ((padding, padding), (padding, padding)), mode='constant')

    m, n = img.shape
    p, q = kernel.shape

    result_m = math.ceil((m - p + 2*padding) / stride)
    result_n = math.ceil((n - q + 2*padding) / stride)

    result = np.empty((result_m, result_n), dtype=int)

    # Convolution
    for i in range(0, m - p + 1, stride):
        for j in range(0, n - q + 1, stride):
            result[i][j] = np.sum(img_padded[i:p+i, j:q+j] * kernel)

    
    return result

# Teste

img = Image.open("../imgs/puc.png")
img = img.convert("L")
img = np.array(img)

kernel = laplace_gauss

m = convolve(img, kernel, stride=1, padding=1)
m = pl.maxpooling(img, (2, 2), 2)

final = Image.fromarray(m)
final.show()