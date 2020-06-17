import numpy as np
import math
from skimage.morphology import disk, dilation


def gen_phi(img_shape, type='whole'):

    p = img_shape[0]
    q = img_shape[1]

    if type == 'whole':
        r = 3
        size = int(round(math.ceil(max(p, q) / 2 / (r + 1)) * 3 * (r + 1)))
        m = np.zeros([size, size])

        h_sz = int(round(size / 2))

        i = np.arange(1, int(round(size / 2 / (r + 1))) + 1)
        j = np.arange(1, int(round(0.9 * size / 2 / (r + 1))) + 2)
        j -= int(round(np.median(j)))

        list_x = h_sz + 2 * j * (r + 1) - 1
        list_y = (2 * i - 1) * (r + 1) - 1

        for x in list_x:
            for y in list_y:
                m[x, y] = 1

        selem = disk(r)
        m = dilation(m, selem)

        m = m[0:p, 0:q]

    return m
