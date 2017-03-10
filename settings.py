# #This file holds all the setting for the project

default_threshold = 200
use_default_threshold = False

k_neighborhood = 7
use_weighted_k_neighborhood = True

use_dilation_for_features = False
use_dilation_for_labeling = False

save_box_train = False
save_box_test = False
save_histogram_train = False

save_distance_matrix_train = False
save_confusion_matrix = False

print_percentage = True

use_filter = False
save_filter = False
filter_size = 20

min_component_size = 10
max_component_size = 100

training_data_dir = './/training_data//'
training_data_visualization_dir='.//training_visualization//'

testing_visualization = './/testing_visualization//'


def set_threshold_mode(is_default):
    global use_default_threshold
    use_default_threshold = is_default


def set_k_neighborhood(is_weighted, size):
    global k_neighborhood, use_weighted_k_neighborhood
    k_neighborhood = size
    use_weighted_k_neighborhood = is_weighted


def set_morphology(for_labeling, for_features):
    global use_dilation_for_features, use_dilation_for_labeling
    use_dilation_for_labeling = for_labeling
    use_dilation_for_features = for_features


def set_threshold(threshold):
    global default_threshold
    default_threshold = threshold


def set_basic_settings():
    set_threshold(200)
    set_threshold_mode(True)
    set_morphology(False, False)
    set_k_neighborhood(False, 1)


def set_enhanced_settings():
    set_threshold_mode(False)
    set_morphology(True, False)
    set_k_neighborhood(True, 7)


def set_enhanced_settings_for_filter():
    global use_filter
    set_threshold_mode(False)
    set_morphology(True, True)
    set_k_neighborhood(True, 7)
    use_filter = True
