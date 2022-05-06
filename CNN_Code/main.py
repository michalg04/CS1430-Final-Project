#!/usr/bin/python
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from helpers import get_image_paths
from cnn import  cnn_classify, get_images
from create_results_webpage import create_results_webpage
from sklearn.metrics import classification_report


def projPenuomniaBoW(feature='placeholder', classifier='placeholder', 
                    data_path='../chest_xray/'):


    # Step 0: Set up parameters, category list, and image paths.
    FEATURE = feature
    CLASSIFIER = classifier

    # This is the list of categories / directories to use. The categories are
    # somewhat sorted by similarity so that the confusion matrix looks more
    # structured (indoor and then urban and then rural).
    categories = ['NORMAL', 'PNEUMONIA']

    # This list of shortened category names is used later for visualization.
    abbr_categories = ['N', 'P']

    # Number of training examples per category to use. Max is 100. For
    # simplicity, we assume this is the number of test cases per category as
    # well.
    num_train_per_cat = 200
    


    print('Getting paths and labels for all train and test data.')
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(data_path, categories, num_train_per_cat)

    print('Using %s representation for images.' % FEATURE)


        
    if FEATURE.lower() == 'none':

        # YOU CODE get_tiny_images (see student.py)
        train_image_feats = get_images(train_image_paths)
        test_image_feats  = get_images(test_image_paths)

    

    
    elif FEATURE.lower() == 'placeholder':
        train_image_feats = []
        test_image_feats = []

    else:
        raise ValueError('Unknown feature type!')

    ############################################################################
    ## Step 2: Classify each test image by training and using the appropriate classifier
    # Each function to classify test features will return an N x 1 string array,
    # where N is the number of test cases and each entry is a string indicating
    # the predicted category for each test image. Each entry in
    # 'predicted_categories' must be one of the 15 strings in 'categories',
    # 'train_labels', and 'test_labels'. See the starter code for each function
    # for more details.
    ############################################################################

    print('Using %s classifier to predict test set categories.' % CLASSIFIER)


    if CLASSIFIER.lower() == 'cnn':
        # YOU CODE svm_classify (see student.py)
        history, predicted_categories = cnn_classify(train_image_feats, train_labels, test_image_feats)

    elif CLASSIFIER.lower() == 'placeholder':
        #The placeholder classifier simply predicts a random category for every test case
        random_permutation = np.random.permutation(len(test_labels))
        predicted_categories = [test_labels[i] for i in random_permutation]

    else:
        raise ValueError('Unknown classifier type')

    
    predicted_labels= np.argmax(predicted_categories, axis =1)
    test_labels = np.array(test_labels)
    #print(predicted_labels)
    #print(test_labels)
    cr = classification_report(test_labels, predicted_labels, output_dict=True)
    print(cr)
    

    acc = accuracy_score(test_labels, predicted_labels)
    print("Accuracy: " + str(acc))
    print("TRAIN ACC")
    print(history.history['sparse_categorical_accuracy'])
    print("VAL ACC")
    print(history.history['val_sparse_categorical_accuracy'])
    print("TRAIN LOSS")
    print(history.history['loss'])
    print("VAL LOSS")
    print(history.history['val_loss'])
    
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.savefig('SNP_accuracy_' +  str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.show()
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.savefig('SNP_loss_' +  str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.show()
    

if __name__ == '__main__':

    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--feature', default='placeholder', help='Either placeholder or none')
    parser.add_argument('-c', '--classifier', default='placeholder', help='Either placeholder or cnn')
    parser.add_argument('-d', '--data', default='../chest_xray', help='Filepath to the data directory')

    args = parser.parse_args()
    projPenuomniaBoW(args.feature, args.classifier, args.data)
