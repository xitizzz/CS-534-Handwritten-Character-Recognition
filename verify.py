import numpy as np
import pickle as pk
from scipy.spatial.distance import cdist

import settings


def verify_ans(predicted_val, component_coord, pkl_path):

    pkl_file = open(pkl_path)
    data = pk.load(pkl_file)
    pkl_file.close()
    classes = data['classes']
    locations = data['locations']

    if len(classes) != len(predicted_val):
        print "The number of character detected is not accurate this might have a minor effect on recognition rate"
        print "Number of characters detected: " + str(len(predicted_val))
        print "Actual number of characters" + str(len(classes))

    distance_matrix = cdist(component_coord, locations)
    dist_index = np.argsort(distance_matrix, axis=1)
    dist_index = dist_index.transpose()

    match = 0
    count = 0

    for i in dist_index[0]:
        if str(predicted_val[count]).lower() == str(classes[i]).lower():
            match += 1
        count += 1

    if settings.print_percentage:
        print "Testing Results"
        print "Number of characters: " + str(count)
        print "Number of correct recognitions: " + str(match)
        print "Recognition rate: " + str(np.double(match)/np.double(len(classes))*100) + " %"