from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import settings
import pickle


def print_characters(path, component_boundaries, predicted_val):
    img = io.imread(path)
    io.imshow(img)
    axes = plt.gca()
    count = 0

    min_row = 0
    min_col = 1
    max_row = 2
    max_col = 3

    for c in component_boundaries:
        axes.add_patch(
            Rectangle((c[min_col]-2, c[min_row]-2), c[max_col] - c[min_col]+4, c[max_row] - c[min_row]+4, fill=False, edgecolor='red',
                      linewidth=1))
        plt.text(c[min_col]-5, c[min_row]-5, predicted_val[count], color='blue', size=10)
        count+=1

    plt.savefig("Output.tiff")
    plt.show()


def pkl_dump(locations, classes):
    pkl_file = open("output.pkl", "w+")
    pickle.dump(locations, pkl_file)
    pickle.dump(classes, pkl_file)
