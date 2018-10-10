import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import numpy as np


def plot(filename, image_data, rows, cols):
    f, axarr = plt.subplots(rows, cols)
    # handle the case when rows == cols == 1
    if type(axarr) is not np.ndarray: axarr = np.array([axarr]).reshape([1,1])
    image_data = image_data.astype(float)
    try:
        for i in range(rows):
            for j in range(cols):
                img = np.squeeze(image_data[i * cols + j])
                axarr[i, j].imshow(img)
                plt.box(False)
                axarr[i, j].axis('off')
    except IndexError:
        pass
    f.savefig(filename)
    plt.close(f)

def plot_single(filename, img):
    im = Image.fromarray(np.squeeze(np.int8(img*255.)), mode='L')
    im.save(filename + '.png')
