from sklearn.decomposition import PCA
from operator import itemgetter
from collections import Counter
import matplotlib.pyplot as plt 
import numpy as np
import random
import scikitplot as skplt

import read_yale

def distance(a, b):
    ''' Calculate the Euclidean distance between data points a and b
        
        @param    a   list of attribute values
        @param    b   list of attribute values

        @return   distance between a and b or -1 if lists have different lenghts 
    '''
    len_a = len(a)
    len_b = len(b)

    if len_a != len_b:
        return(-1)
    else:
        dist = 0
        for i in range(0, len_a):
            dist += (a[i] - b[i]) ** 2

        return np.sqrt(dist)


def compute_distances(test_data, training_data):
    ''' Compute distances from each data point in training data to that in test data

        @param    training_data  list of numerical data rows, excluding the attribute to predict
        @param    test_data      list of the data object to calculate distances to

        @return   distance list
    '''

    distances = []
    
    for i in range(0, len(test_data)):
        row = []
        for j in range(0, len(training_data)):
            dist = distance(test_data[i], training_data[j])
            row.append(dist)

        distances.append(row)

    return(distances)


def compute_nearest_neighbors(test_data, training_data, num_neighbors):
    ''' Compute nearest neighbors of the given data object

        @param    test_data       data object to compute distances to
        @param    training_data   data that the distances are computed to
        @param    num_neighbors   value of k

        @return   Indices of num_neighbors nearest neighbors of test_data in training_data
    '''

    #Compute distances to test_data
    distances = compute_distances(test_data, training_data)
    neighbors = []
    for ix_test_obj in range(0, len(test_data)):
        #Iterate over the distance list and create list of (row index, distance) pairs
        ixs_and_distances = []
        for i in range(0, len(distances[ix_test_obj])):
            ixs_and_distances.append([ i, distances[ix_test_obj][i] ])

        #Sort the list on distances and include only first num_neighbors elements
        ixs_and_distances.sort(key = itemgetter(1))
        k_nearest = ixs_and_distances[0 : num_neighbors]

        #Only return neighbor indices in training_data
        neighbors_of_row = []
        for n in k_nearest:
            neighbors_of_row.append(n[0])

        neighbors.append(neighbors_of_row)

    return(neighbors)

def majority_class(neighbor_ixs, classes):
    ''' Get the majority class in list neighbors

        @param    neighbor_ixs   list of neighbor indices
        @param    classes        list of the correct classes in the training data

        @return   value of the majority class 
    '''
    
    neighbor_classes = []
    for n in neighbor_ixs:
        neighbor_classes.append(classes[n])
    
    c = Counter(neighbor_classes)
    return(c.most_common()[0][0])


def save_png(pixels, width, height):
    plt.imshow(pixels.reshape(height, width), cmap=plt.cm.bone)
    plt.savefig("img.png")
    plt.show()


def get_test_ixs(labels, num_tests_per_subject):
    '''
    Get random indices of pictures that will be used in the test set
    
    @param labels                   List of label names of all pictures
    @param num_tests_per_subject    How many pictures to include per subject

    @return list of indices to use in the test set
    '''

    test_ixs = []
    lbl = ""
    i = 0
    while i < len(labels) - 1:
        lbl = labels[i]

        how_many_pics = 0
        while (i + how_many_pics + 1) < len(labels) and lbl == labels[i + how_many_pics]:
            how_many_pics = how_many_pics + 1

        low_bound = i
        high_bound = i + how_many_pics

        for j in range(num_tests_per_subject):
            ix = random.randint(low_bound, high_bound)
            test_ixs.append(ix)    
        
        i = high_bound + 1
    
    return test_ixs


#Load data set and extract images and labels
a = read_yale.load_images_croppedyale() #tuple: pics, fnames, lnames
pics = a[0]
suffixes = a[2]
imgs = read_yale.images_to_array(pics, suffixes)

pixels = imgs[0]
labels = imgs[1]

#Get image indices that will be used in the test set
num_tests_per_subject = 4
test_ixs = get_test_ixs(labels, num_tests_per_subject)


predictions = []
actuals = []
correct = 0
incorrect = 0
for test_ix in test_ixs:
    print("test ix: " + str(test_ix))

    #Split to test and training sets
    test_pixels = pixels[test_ix]
    training_pixels = pixels
    np.delete(training_pixels, test_ix)
    test_labels = labels[test_ix]
    training_labels = labels
    np.delete(training_labels, test_ix)
    
    #Create PCA base from training set 
    pca = PCA(n_components = 150)
    pca.fit(training_pixels)
    eigenfaces = pca.components_

    IMG_WIDTH = 168
    IMG_HEIGHT = 192

    #Transform images
    training_trf = pca.transform(training_pixels)
    test_trf = pca.transform([test_pixels])

    #Compute nearest neighbors and choose majority label as the prediction
    num_neighbors = 5   
    neighbors = compute_nearest_neighbors(test_trf, training_trf, num_neighbors)
    prediction = majority_class(neighbors[0], training_labels.tolist())
    
    print("prediction: " + prediction + ", actual: " + test_labels)
    if test_labels == prediction:
        correct = correct + 1
    else:
        incorrect = incorrect + 1
    
    predictions.append(prediction)
    actuals.append(test_labels)

#Some ad-hoc cropping of the names to fit in the confusion matrix
cropped_actuals = []
cropped_predictions = []
for act in actuals:
    cropped_actuals.append(act[5:])
for pred in predictions:
    cropped_predictions.append(pred[5:])

#Plot a confusion matrix
skplt.metrics.plot_confusion_matrix(cropped_actuals, cropped_predictions, text_fontsize='small',
    hide_zeros=True, normalize=True, figsize=(15,15))
plt.savefig("confusion_eigenfaces.png")
plt.gcf().clear()

print("Correct " + str(correct) + "/" + str(correct + incorrect))
