from math import sqrt
from operator import itemgetter

# Calculate the Euclidean distance between data points a and b
#
# @param    a   list of attribute values
# @param    b   list of attribute values
#
# @return   distance between a and b or -1 if lists have different lenghts 
def distance(a, b):
    len_a = len(a)
    len_b = len(b)
    
    if len_a != len_b:
        return(-1)
    else:
        dist = 0
        for i in range(0, len_a):
            dist += (a[i] - b[i]) ** 2

        return sqrt(dist)

# Compute distances from each data point in training data to that in test data
#
# @param    training_data  list of numerical data rows, excluding the attribute to predict
# @param    test_data      list of the data object to calculate distances to
#
# @return   distance list
def compute_distances(test_data, training_data):
    num_rows = len(training_data)

    distances = []

    for i in range(0, num_rows):
        dist = distance(test_data, training_data[i])
        distances.append(dist)
        
    return(distances)

# Compute nearest neighbors of the given data object
#
# @param    test_data       data object to compute distances to
# @param    training_data   data that the distances are computed to
# @param    num_neighbors   value of k
#
# @return   num_neighbors nearest neighbors of test_data in training_data
def compute_nearest_neighbors(test_data, training_data, num_neighbors):
    #Compute distances to test_data
    distances = compute_distances(test_data, training_data)
    
    #Iterate over the distance list and create list of (row index, distance) pairs
    ixs_and_distances = []
    for i in range(0, len(distances)):
        ixs_and_distances.append([i, distances[i]])

    #Sort the list on distances and return first num_neighbors elements
    ixs_and_distances.sort(key = itemgetter(1))
    return(ixs_and_distances[0 : num_neighbors])

# Get the majority class in list neighbors
#
# @param    neighbors   list of (index, distance) pairs
# @param    classes     list of the correct classes in the training data
#
# @return   value of the majority class 
def majority_class(neighbors, classes):
    neighbor_classes = []
    for n in neighbors:
        neighbor_classes.append(classes[n[0]])

    return(max(set(neighbor_classes), key = neighbor_classes.count))

# Predict the class of test data object from it's nearest neighbors in training data
#
# @param    test_data       the data object whose class is predicted
# @param    training_data   list of data objects where neighbors are searched
# @param    target_list     correct classes of objects in training data
# @param    k               number of nearest neighbors to search
#  
# @return   value of the predicted class
def predict_class(test_data, training_data, target_list, k):
    
    neighbors = compute_nearest_neighbors(test_data, training_data, k)
    predicted_class = majority_class(neighbors, target_list)
    
    return(predicted_class)

# Perform the k-nearest-neighbors classification with leave-one-out
# cross-validiation on data
#
# @param    data        list of the data objects to analyze
# @param    ix_target   index of the field whose value will be predicted
# @param    k           number of neighbors to search
#
# @return   misclassification rate of the classification
def classification_with_cross_validation(data, ix_target, k):
    data_without_targets = [[0] * len(data[0])] * len(data)
    rows = len(data_without_targets)

    #Delete the target attribute from data
    classes = []
    for i in range(0, rows):
        classes.append(data[i][ix_target])
        data_without_targets[i] = data[i][0 : ix_target]

    #Perform predictions and count misclassifications
    misclassifications = 0
    for i in range(0, rows):
        
        #Split data into single test object and training data
        test_data = data_without_targets[i]
        training_data = data_without_targets[:i] + data_without_targets[i + 1 : rows]

        #Predict a class for the test data
        predicted_class = predict_class(test_data, training_data, classes, k)

        if(predicted_class != classes[i]):
            misclassifications += 1
    
    #Calculate and return the misclassification rate 
    misclassification_rate = float(misclassifications) / rows
    return misclassification_rate
