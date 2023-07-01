import numpy as np
import math

def maxpooling(img: np.array, pool_size: tuple, stride: int) -> np.array:
    m, n = img.shape
    p, q = pool_size
    
    result_m = math.ceil(((m - p) / stride) + 1)
    result_n = math.ceil(((n - q) / stride) + 1)
    
    result = np.empty((result_m, result_n), dtype=int)
    
    for i in range(0, m - p + 1, stride):
        for j in range(0, n - q + 1, stride):
            result[i//stride][j//stride] = np.max(img[i:p+i, j:q+j])
            
    return result

if __name__ == '__main__':

    img = np.array([[8, 1, 3, 6], 
                    [3, 2, 2, 1],
                    [5, 0, 7, 1],
                    [2, 4, 9, 7]])

    pool_size = (2, 2)
    stride = 2

    print(maxpooling(img, pool_size, stride))