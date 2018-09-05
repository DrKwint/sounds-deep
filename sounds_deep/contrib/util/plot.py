import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot(filename, image_data, rows, cols):
    f, axarr = plt.subplots(rows,cols)
    try:
        for i in range(rows):
            for j in range(cols):
                axarr[i, j].imshow(image_data[i*cols + j])
                axarr[i,j].axis('off')
    except IndexError:
        pass
    f.savefig(filename)
    plt.close(f)
