from math import sqrt
from operator import itemgetter
from numpy import mean
from collections import Counter


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

        return sqrt(dist)


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


def predict_regression(test_inputs, training_inputs, training_outputs, k):
    ''' Predict an output for an input based it's the nearest neighbors

        @param    test_inputs         Inputs to predict an output for
        @param    training_inputs     Data to calculate neighbors from
        @param    training_outputs    True outputs for training_inputs
        @param    k                   Number of neighbors to use

        @return   Mean value of k nearest neighbors
    '''
    neighbor_ixs = compute_nearest_neighbors(test_inputs, training_inputs, k)
        
    neighbor_outputs = []
    for i in neighbor_ixs[0]:
        neighbor_outputs.append(training_outputs[i])

    prediction = mean(neighbor_outputs)

    return(prediction)


def majority_class(neighbor_ixs, classes):
    ''' Get the majority class in list neighbors

        @param    neighbor_ixs   list of neighbor indices
        @param    classes        list of the correct classes in the training data

        @return   value of the majority class 
    '''
    
    neighbor_classes = []
    for n in neighbor_ixs:
        neighbor_classes.append(classes[n][0])
    
    c = Counter(neighbor_classes)
    return(c.most_common()[0][0])


def predict_classification(test_data, training_data, target_list, k):
    ''' Predict the class of test data object from it's nearest neighbors in training data

        @param    test_data       the data object whose class is predicted
        @param    training_data   list of data objects where neighbors are searched
        @param    target_list     correct classes of objects in training data
        @param    k               number of nearest neighbors to search
        
        @return   value of the predicted class
    '''

    neighbors = compute_nearest_neighbors(test_data, training_data, k)

    predictions = [] 
    for ix_test in range(0, len(test_data)):
        predicted_class = majority_class(neighbors[ix_test], target_list)
        predictions.append(predicted_class)

    return(predicted_class)


def c_index(true_values, predicted_values):
    ''' Compute C-index for list of predictions

        @param    true_values         True target values for the data
        @param    predicted_values    Predicted target values
    '''

    n = 0
    h_sum = 0.0

    for i in range(0, len(true_values)):
        t = true_values[i]
        p = predicted_values[i]

        for j in range(i + 1, len(true_values)):
            nt = true_values[j]
            np = predicted_values[j]

            if (t != nt):
                n += 1

                if ((p < np and t < nt) or (p > np and t > nt)):
                    h_sum += 1
                elif ((p < np and t > nt) or (p > np and t < np)):
                    h_sum += 0
                elif (p == np):
                    h_sum += 0.5

    c_idx = h_sum / n
    return c_idx