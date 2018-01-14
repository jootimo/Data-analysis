import math
from operator import itemgetter

# Calculate the Euclidean distance between data points a and b
def distance(a, b):
    len_a = len(a)
    len_b = len(b)
    
    if len_a != len_b:
        return(-1)
    else:
        dist = 0
        for i in range(0, len_a):
            dist += (a[i] - b[i]) ** 2

        return math.sqrt(dist)

# Compute distances from each data point to another
#
# @param    data_matrix  list of numerical data rows, excluding the value to predict
# @return   distance matrix
def compute_distances(data_matrix):
    num_rows = len(data_matrix)

    distances = []

    for i in range(0, num_rows):

        row = []
        for j in range(0, num_rows):
            row.append(distance(data_matrix[i], data_matrix[j]))
        
        distances.append(row)
        
    return(distances)



def compute_nearest_neighbors(distance_matrix, ix_row, num_neighbours):
    #Iterate over the distance matrix and create list of row index-distance pairs
    ixs_and_distances = []
    for i in range(0, len(distance_matrix[0])):
        
        #Don't include distance to self
        if i == ix_row:
            continue

        ixs_and_distances.append( [i, distance_matrix[ix_row][i]] )

    #Sort the list on distances and return first num_neighbors elements
    ixs_and_distances.sort(key = itemgetter(1))
    return(ixs_and_distances[0 : num_neighbours])



def majority_class(neighbors, classes):
    neighbor_classes = []
    for n in neighbors:
        neighbor_classes.append(classes[n[0]])

    return(max(set(neighbor_classes), key = neighbor_classes.count))

def predict_class(data_matrix, target_list, i, k):
    distances = compute_distances(data_matrix)
    neighbors = compute_nearest_neighbors(distances, i, k)
    predicted_class = majority_class(neighbors, target_list)
    
    return(predicted_class)

def classification_with_cross_validation(data, ix_target, k):
    data_without_targets = [[0] * len(data[0])] * len(data)
    classes = []
    rows = len(data_without_targets)
    for i in range(0, rows):
        classes.append(data[i][ix_target])
        data_without_targets[i] = data[i][0 : ix_target]

    misclassifications = 0
    for i in range(0, rows):
        distance_matrix = compute_distances(data_without_targets)
        predicted_class = predict_class(data_without_targets, classes, i, k)

        if(predicted_class != classes[i]):
            misclassifications += 1
    
    misclassification_rate = misclassifications / rows
    return misclassification_rate