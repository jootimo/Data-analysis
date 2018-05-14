import knn                      # K-nearest-neighbors utilities
from functools import partial   # Function argument binding
from numpy import ndarray       # array -> list conversion
from scipy import stats         # z-score

DATA_FILENAME = "data/painsignals.csv"

IX_SUBJECT      = 0
IX_TEST         = 1
IX_MEAS_ID      = 2
IX_HR           = 3
IX_RRPM         = 4
IX_GSR          = 5
IX_RMSCORR      = 6
IX_RMSORB       = 7
IX_LABEL        = 8

NUM_NUMERIC_FEATURES = 5

# Open file with name filename and parse comma separated values a into 2-d list
#
# @param    filename
def open_and_parse(filename):
    data = []

    with open(filename, "r") as filestream:

        for line in iter(filestream.readline, ''):
            line = list(line.split(','))

            #First row is headers, skip it
            if line[0] == "subject":
                continue

            #Parse rows into lists
            for i in range(0, len(line)):
                for char in range(0, len(line[i])):

                    #Remove newline characters
                    if type(line[i]) == "str":
                        line[i].replace('\n', '')

                    #Cast to correct type
                    if i in [IX_SUBJECT, IX_MEAS_ID, IX_TEST, IX_LABEL]:
                        line[i] = int(line[i])
                    else:
                        line[i] = float(line[i])

            data.append(line)

    return data

# Get the data that belongs to a single subject 
#
# @param    subject_id  Id of the subject to get data of
# @param    data        2-d data list
#
# @return   Rows in data belonging to subject with id subject_id
def get_rows_of_subject(subject_id, data):
    rows = []
    for row in data:
        if(row[IX_SUBJECT] == subject_id):
            rows.append(row)

    return rows

# Get the data that belongs to other subjects than the specified subject
#
# @param    subject_id  The subject NOT to get data of
# @param    data        2-d data list
#
# @return   Rows in data that don't belong to subject with id subject_id
def get_rows_of_other_subjects(subject_id, data):
    rows = []
    for row in data:
        if(row[IX_SUBJECT] != subject_id):
            rows.append(row)

    return rows

# Get the ids of subjects in data
#
# @param    data    2-d data list
#
# @reuturn  List of subject ids in data  
def get_subject_ids(data):
    subject_ids = []
    prev_id = -1
    for row in data:
        current_id = row[IX_SUBJECT]
        if current_id != prev_id:
            subject_ids.append(int(current_id))
            prev_id = current_id

    return subject_ids


# Get the values of hr, rrpm, gsr, rmscorr, rmsorb from data
#
# @param    data    2-d data list
#
# @return   2-d list containing only the feature values
def get_features(data):
    features = []
    for row in data:
        features.append(row[IX_HR : IX_RMSORB + 1])

    return features

# Get the pain labels from data
#
# @param    data    2-d data list
#
# @return   2-d list containing only the label values
def get_labels(data):
    labels = []
    for row in data:
        labels.append(row[IX_LABEL])

    return labels


# Standardize a list with z-score
# 
# @param    data    The list to standardize
#
# @return   The standardized data
def z_score_list(data):
    standardized = ndarray.tolist(stats.zscore(data.copy()))
    return standardized


# Standardize the features of each subject independently
#
# @param    data                2-d data list
# @param    ix_features_start   Index in data where the features start
# @param    num_features        Number of features in data
# @param    f_standardize       Function to standardize with
#
# @return   Per-subject standardized data
def standardize_features_per_subject(data, ix_features_start, num_features, f_standardize):
    data_standardized = []
    subject_ids = get_subject_ids(data)

    for id in subject_ids:
        rows = get_rows_of_subject(id, data)
        features = get_features(rows)
        
        #Standardize rows of this subject
        features_standardized = f_standardize(features)

        for row_ix in range(0, len(rows)):
            #Place the standardized values to the correct place 
            #in the row 
            row_concat = rows[row_ix][0:ix_features_start]

            for feat in features_standardized[row_ix]:
                row_concat.append(feat)

            row_concat = (row_concat 
                + rows[row_ix][ix_features_start + num_features : len(rows[row_ix])])

            data_standardized.append(row_concat)

    return data_standardized


# Standardize the data
#
# @param    data                2-d data list
# @param    ix_features_start   Index in data where the features start
# @param    num_features        Number of features in data
# @param    f_standardize       Function to standardize with
#
# @return   Standardized data
def standardize_features(data, ix_features_start, num_features, f_standardize):
    data_standardized = []

    features = get_features(data)
    features_standardized = f_standardize(features)

    for row_ix in range(0, len(data)):
        #Place the standardized values to the correct place 
        #in the row 
        row_concat = data[row_ix][0:ix_features_start]

        for feat in features_standardized[row_ix]:
            row_concat.append(feat)

        row_concat = (row_concat 
            + data[row_ix][ix_features_start + num_features : len(data[row_ix])])
        data_standardized.append(row_concat)

    return data_standardized


# Perform k-fold cross-validation where each fold contains only data from
# a single subject
#
# @param    data_matrix     2-d data list 
# @param    num_neighbors   The number of neighbors used in knn
# @param    f_standardize   Function used for standardizing the data
def cross_validate_per_subject(data_matrix, f_standardize, f_predict):

    data = standardize_features_per_subject(
        data_matrix, IX_HR, NUM_NUMERIC_FEATURES, f_standardize)

    subject_ids = get_subject_ids(data)

    c_ixs = []

    for subj_id in subject_ids:
        test_data = get_rows_of_subject(subj_id, data)
        training_data = get_rows_of_other_subjects(subj_id, data)

        c_ixs.append(f_predict(test_data, training_data, subj_id))

    print("Average C-index: " + str(sum(c_ixs) / float(len(c_ixs))))


# Perform k-nearest-neighbors classification for test data
# and get the c-index
#
# @param    test_data       Data to predict labels of
# @param    trainining_data Data to calculate neighbors from
# @param    subject_id      Id of the subject the test data belongs to. Only used for printing
# @param    k               The number of neighbors to search
#
# @return   c-index for predictions
def knn_classification_and_c_index(test_data, training_data, subject_id, k):
    test_labels = get_labels(test_data)
    training_labels = get_labels(training_data)

    test_features = get_features(test_data)
    training_features = get_features(training_data)

    # Get nearest neighbors for every object in test set
    neighbors = knn.compute_nearest_neighbors(test_features, training_features, k)

    predictions = []
    actuals = []
    for measurement in range(0, len(neighbors)):
        prediction = knn.majority_class(neighbors[measurement], training_labels)
        actual = test_labels[measurement]

        predictions.append(prediction)
        actuals.append(actual)

    c_ix = knn.c_index(actuals, predictions)
    print("(" + str(subject_id) + "," + str(c_ix) + ")")

    return(c_ix)


####################################
########## Actual script ###########
####################################

DATA = open_and_parse(DATA_FILENAME)

#Bind the argument 'k' to function knn_classification_and_c_index
NUM_NEIGHBORS = 37
F_PREDICT = partial(knn_classification_and_c_index, k=NUM_NEIGHBORS)

cross_validate_per_subject(DATA, z_score_list, F_PREDICT)
