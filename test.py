import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
from skimage.measure import label, regionprops, moments, moments_central, moments_hu, moments_normalized, perimeter
from skimage import io, transform, morphology, filters
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import settings
import verify
import output
import train


test_features = []
component_centers = []
component_boundaries = []
visualize = False
count = 0
regions = []
img_bin = []
predicted_val_filter = []


def buildFeatures(path, img_name):
    global count
    global regions, img_bin

    img = io.imread(path);

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

    if settings.save_box_test:
        io.imshow(img_bin)
        axes = plt.gca()

    for props in regions:
        min_row, min_col, max_row, max_col = props.bbox
        if max_col - min_col > settings.min_component_size and max_row - min_row > settings.min_component_size and max_col - min_col < settings.max_component_size and max_row - min_row < settings.max_component_size:
            if settings.save_box_test:
                axes.add_patch(
                    Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, fill=False, edgecolor='red',
                              linewidth=1))
                plt.text(min_col, min_row, str(count), color='white', size=8)

            component_boundaries.append([min_row, min_col, max_row, max_col])
            component_centers.append([(min_col + max_col) / 2, (min_row + max_row) / 2])
            component = img_bin[min_row:max_row, min_col:max_col]
            resized_component = transform.resize(img_bin[min_row:max_row, min_col:max_col], (settings.filter_size, settings.filter_size))

            max_val = 0
            max_i = 0

            for i in range(16):
                if np.sum(np.multiply(train.custom_filters[i], resized_component)) > max_val:
                    max_val = np.sum(np.multiply(train.custom_filters[i], resized_component))
                    max_i = i

            predicted_val_filter.append(train.filter_label[max_i])

            moment = moments(component)
            cr = moment[0, 1]/moment[0, 0]
            cc = moment[1, 0]/moment[0, 0]
            mu = moments_central(component, cr, cc)
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            test_features.append(hu)
            count += 1

    if settings.save_box_test:
        plt.savefig(settings.testing_visualization+img_name+".tiff")
        plt.close()


def test_data(image_path, pkl_path):
    global test_features

    buildFeatures(image_path, "Boxed_Image")

    test_features = (test_features - train.mean) / train.sd

    distance_matrix = cdist(test_features, train.features)
    dist_index = np.argsort(distance_matrix, axis=1)
    dist_index = dist_index.transpose()

    label_index = [[train.class_label[x] for x in a] for a in dist_index[0:settings.k_neighborhood, :]]
    if settings.use_weighted_k_neighborhood:
        for i in range(settings.k_neighborhood):
            for j in range(i):
                label_index.append(label_index[settings.k_neighborhood - i - 1])

    mode_class = mode(label_index, nan_policy='raise')
    predicted_val = mode_class[0][0]

    if settings.use_filter:
        verify.verify_ans(predicted_val_filter, component_centers, pkl_path)
    else:
        verify.verify_ans(predicted_val, component_centers, pkl_path)

    if settings.use_filter:
        output.print_characters(image_path, component_boundaries, predicted_val_filter)
        output.pkl_dump(component_centers, predicted_val_filter)
    else:
        output.print_characters(image_path, component_boundaries, predicted_val)
        output.pkl_dump(component_centers, predicted_val)
