import knn
import csv_parser

from functools import partial   # Function argument binding


def loo_cv (features, labels, f_predict):
    '''
    Perform leave-one-out cross-validation, do classifications and report the c-index

    @param  features    2d list of feature values
    @param  labels      Correct labels corresponding to rows in features
    @param  f_predict   Prediction function used for classifications

    @return Concordance-index for the made predictions 
    '''

    predictions = []
    misclassifications = 0
    for test_ix, test_features in enumerate(features):
        training_features = [row for i, row in enumerate(features) if i != test_ix]
        training_labels = [row for i, row in enumerate(labels) if i != test_ix]

        prediction = f_predict([test_features], training_features, training_labels)
        predictions.append(prediction)
        
        #See if we guessed correctly
        actual = labels[test_ix][0]
        if actual != prediction:
            misclassifications += 1    
    

    num_rows = len(features)
    if num_rows != 0:
        misclass_rate = float(misclassifications) / num_rows
        print("Classifications: " + str(num_rows))
        print("Misclassifications: " + str(misclassifications))
        print("Misclassification rate: " + str(misclass_rate))

    c_ix = knn.c_index(labels, predictions)
    return(c_ix)


def filter_list(lst, f_filter):
    '''
    Filter a list with a provided function

    @param  lst         The list to filter
    @param  f_filter    Function that takes a list index as parameter and returns 
                        True if item is accepted to the filtered list
    
    @return A filtered list
    '''

    res = []
    for index, item in enumerate(lst):
        if(f_filter(index)):
            res.append(item)

    return(res)


def pairs_disjoint(ix_0, ix_1, pairs):
    '''
    Check whether pairs of an object clash with those of another object 

    @param  ix_0    Index of one object in pairs list
    @param  ix_1    Index of another object in pairs list
    @param  pairs   List of pair names that are checked for overlapping 
    '''

    pair0 = pairs[ix_0][0]
    pair1 = pairs[ix_0][1]
    other_pair0 = pairs[ix_1][0]
    other_pair1 = pairs[ix_1][1]

    if (pair0 == other_pair0
        or pair0 == other_pair1
        or pair1 == other_pair0
        or pair1 == other_pair1):
        return False;
    
    return True


def loo_cv_with_pairwise_filtering(features, labels, pairs, f_predict):
    '''
    Perform modified leave-one-out cross-validation, where no overlapping of items in
    the pairs list between test and training set is allowed.
    Do classifications and report the c-index.

    @param  features    2d list of feature values
    @param  labels      Correct labels corresponding to rows in features
    @param  pairs       List of pairs of each row in features. These are just names, not eg. row indices 
    @param  f_predict   Prediction function used for classifications

    @return Concordance-index for the made predictions 
    '''

    predictions = []
    misclassifications = 0
    for test_ix, test_features in enumerate(features):
        #Potential training features and labels
        training_features = [row for i, row in enumerate(features) if i != test_ix]
        training_labels = [row for i, row in enumerate(labels) if i != test_ix]
        
        #Bind test index and pairs list to a function that tells whether or not  
        #the pairs of a training object clash with the pairs of this test object  
        f_can_add_to_trn_set = partial(pairs_disjoint, pairs = pairs, ix_1 = test_ix)
        
        #Filter the features and labels
        training_features_filtered = filter_list(training_features, f_can_add_to_trn_set)
        training_labels_filtered = filter_list(training_labels, f_can_add_to_trn_set)

        #Do prediction with the function that was given as an argument
        prediction = f_predict([test_features], training_features_filtered, training_labels_filtered)
        predictions.append(prediction)

        #See if we guessed correctly
        actual = labels[test_ix][0]
        if actual != prediction:
            misclassifications += 1    
    

    num_rows = len(features)
    if num_rows != 0:
        misclass_rate = float(misclassifications) / num_rows
        print("Classifications: " + str(num_rows))
        print("Misclassifications: " + str(misclassifications))
        print("Misclassification rate: " + str(misclass_rate))

    return(knn.c_index(labels, predictions))
        

#######################
#    Actual script    #
#######################

FILENAME_FEATURES = "data/symmetric_pair_input/features.data"
FILENAME_LABELS = "data/symmetric_pair_input/labels.data"
FILENAME_PAIRS = "data/symmetric_pair_input/pairs.data"

features = csv_parser.parse(FILENAME_FEATURES)
labels = csv_parser.parse(FILENAME_LABELS, int)
pairs = csv_parser.parse(FILENAME_PAIRS, str)

#Bind k to the classification function so that cross-validation doesn't need 
#to know anything about the prediction function
num_neighbors = 1
f_predict = partial(knn.predict_classification, k=num_neighbors)

#First perform normal leave-one-out cross-validation
c_ix = loo_cv(features, labels, f_predict)
print("c_index for loo_cv and knn with k = " + str(num_neighbors) + " was " + str(c_ix) + "\n")

#Then the modified cross-validation that considers the protein pairs
c_ix = loo_cv_with_pairwise_filtering(features, labels, pairs, f_predict)
print("c_index for modified loo_cv and knn with k = " + str(num_neighbors) + " was " + str(c_ix))