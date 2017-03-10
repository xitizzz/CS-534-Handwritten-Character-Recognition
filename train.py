import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from scipy.stats import mode
from skimage.measure import label, regionprops, moments, moments_central, moments_hu, moments_normalized
from skimage import io, exposure, transform, morphology, filters
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import settings

features = []
class_label = []
visualize = True
mean = []
sd = []
threshold = 200
custom_filters = []
filter_label =[]


def buildFeatures(path, c):
    global threshold
    img = io.imread(path)

    if settings.save_histogram_train:
        hist = exposure.histogram(img)
        plt.bar(hist[1], hist[0])
        plt.savefig(settings.training_data_visualization_dir+"Histogram_"+c+".tiff")
        plt.close()

    if settings.use_default_threshold:
        threshold = settings.default_threshold
    else:
        threshold = filters.threshold_yen(img)

    img_bin = (img < threshold).astype(np.double)

    if settings.use_dilation_for_labeling or settings.use_dilation_for_features:
        img_bin_dia = morphology.binary_dilation(img_bin, morphology.disk(2)).astype(np.double)

    if settings.use_dilation_for_features:
        img_bin = img_bin_dia

    if settings.use_dilation_for_labeling:
        img_label = label(img_bin_dia, background=0)
    else:
        img_label = label(img_bin, background=0)

    regions = regionprops(img_label)

    if settings.save_box_train:
        io.imshow(img_bin)
        axes = plt.gca()

    count = 0

    custom_filter = np.zeros((settings.filter_size, settings.filter_size), dtype=np.int16)

    for props in regions:
        min_row, min_col, max_row, max_col = props.bbox
        if max_col - min_col > settings.min_component_size and max_row - min_row > settings.min_component_size and max_col - min_col < settings.max_component_size and max_row - min_row < settings.max_component_size:
            if settings.save_box_train:
                axes.add_patch(
                    Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, fill=False, edgecolor='red',
                              linewidth=1))

            component = img_bin[min_row:max_row, min_col:max_col]

            custom_filter = custom_filter + transform.resize(img_bin[min_row:max_row, min_col:max_col], (settings.filter_size, settings.filter_size))

            moment = moments(component)
            cr = moment[0, 1]/moment[0, 0]
            cc = moment[1, 0]/moment[0, 0]
            mu = moments_central(component, cr, cc)
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            features.append(hu)
            class_label.append(c)
            count += 1

    custom_filters.append(normalize(custom_filter))
    filter_label.append(c)

    if settings.save_box_train:
        plt.savefig(settings.training_data_visualization_dir + "Boxed_" + c + ".tiff")
        plt.close()

    if settings.save_filter:
        io.imshow(normalize(custom_filter))
        plt.savefig(settings.training_data_visualization_dir+"Filter_" + c + ".tiff")
        plt.close()


def train_model():
    global features, mean, sd

    characters = ['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z']

    for c in characters:
        buildFeatures(settings.training_data_dir + c + '.bmp', c)

    mean = np.mean(features, axis=0)
    sd = np.std(features, axis=0)

    features = (features-mean)/sd

    distance_matrix = cdist(features, features)

    if settings.save_distance_matrix_train:
        plt.imshow(distance_matrix)
        plt.savefig(settings.training_data_visualization_dir+"Distance_matrix.tiff")
        plt.close()

    dist_index = np.argsort(distance_matrix, axis=1)
    dist_index = dist_index.transpose()

    true_val = [class_label[x] for x in dist_index[0]]
    label_index = [[class_label[x] for x in a] for a in dist_index[1:settings.k_neighborhood + 1, :]]

    if settings.use_weighted_k_neighborhood:
        for i in range(settings.k_neighborhood):
            for j in range(i):
                label_index.append(label_index[settings.k_neighborhood-i-1])

    mode_class = mode(label_index, nan_policy='raise')
    predicted_val = mode_class[0][0]

    conf_mat = confusion_matrix(true_val, predicted_val)

    match = 0

    for i in range(len(true_val)):
        if true_val[i] == predicted_val[i]:
            match += 1

    if settings.print_percentage:
        print "Training set"
        print "Number of Characters: " + str(len(true_val))
        if settings.use_filter:
            print "Recognition rate: " + "N.A."
        else:
            print "Recognition rate: " + str(np.double(np.double(match)/np.double(len(true_val)))*100)+" %"
        print "------------------------"

    if settings.save_confusion_matrix:
        io.imshow(conf_mat)
        plt.savefig(settings.training_data_visualization_dir+"Confusion_matrix.tiff")
        plt.close()
