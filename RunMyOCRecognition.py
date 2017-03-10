# #Optical Character Recognition
# #
# #Give image path in 'img_path' and path to pickle file in 'ground_truth_path'
# #The enhancement can be selected run time
# #Program will print recognition rate and display recognition result
# #Visulizations can be modified in settings.py
# #
# #Kshitij Shah

import settings
import train
import test


def main():

    # #give image path here
    img_path = './/testing_data//test1.bmp'

    # #give ground truth path here
    ground_truth_path = ".//testing_data//test1_gt.pkl"

    wait = True
    while wait:
        choice = input("Press 1 for Basic, 2 for Enhanced and 3 for Enhanced with filter: ")
        if choice == 1:
            settings.set_basic_settings()
            wait = False
        elif choice == 2:
            settings.set_enhanced_settings()
            wait = False
        elif choice == 3:
            settings.set_enhanced_settings_for_filter()
            wait = False
        else:
            print "No kidding"

    train.train_model()

    test.test_data(img_path, ground_truth_path)

if __name__ == '__main__':
    main()
