import itertools

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image


def plot(filename, image_data, rows, cols):
    f, axarr = plt.subplots(rows, cols)
    # handle the case when rows == cols == 1
    if type(axarr) is not np.ndarray: axarr = np.array([axarr]).reshape([1, 1])
    image_data = image_data.astype(float)
    try:
        for i in range(rows):
            for j in range(cols):
                img = np.squeeze(image_data[i * cols + j])
                axarr[i, j].imshow(img)
                axarr[i, j].axis('off')
    except IndexError:
        pass
    f.savefig(filename)
    plt.close(f)


def plot_single(filename, img):
    im = Image.fromarray(np.squeeze(np.int8(img * 255.)), mode='L')
    im.save(filename + '.png')


def plot_confusion_matrix(cm,
                          classes,
                          filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
