import knn
from numpy import ndarray
from scipy import stats

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


def get_rows_of_subject(subject_id, data):
    rows = []
    for row in data:
        if(row[IX_SUBJECT] == subject_id):
            rows.append(row)

    return rows


def get_rows_of_other_subjects(subject_id, data):
    rows = []
    for row in data:
        if(row[IX_SUBJECT] != subject_id):
            rows.append(row)

    return rows


def get_subject_ids(data):
    subject_ids = []
    prev_id = -1
    for row in data:
        current_id = row[IX_SUBJECT]
        if current_id != prev_id:
            subject_ids.append(int(current_id))
            prev_id = current_id

    return subject_ids


def get_features(data):
    features = []
    for row in data:
        features.append(row[IX_HR : IX_RMSORB + 1])

    return features


def get_labels(data):
    labels = []
    for row in data:
        labels.append(row[IX_LABEL])

    return labels


def standardize_features_per_subject(data, ix_features_start, num_features, f_standardize):
    data_standardized = []
    subject_ids = get_subject_ids(data)

    for id in subject_ids:
        rows = get_rows_of_subject(id, data)
        features = get_features(rows)
        features_standardized = f_standardize(features)

        for row_ix in range(0, len(rows)):
            row_concat = rows[row_ix][0:ix_features_start]

            for feat in features_standardized[row_ix]:
                row_concat.append(feat)

            row_concat = row_concat + rows[row_ix][ix_features_start + num_features : len(rows[row_ix])]
            data_standardized.append(row_concat)

    return data_standardized


def standardize_features(data, ix_features_start, num_features, f_standardize):
    data_standardized = []

    features = get_features(data)
    features_standardized = f_standardize(features)


    for row_ix in range(0, len(data)):
        row_concat = data[row_ix][0:ix_features_start]

        for feat in features_standardized[row_ix]:
            row_concat.append(feat)

        row_concat = row_concat + data[row_ix][ix_features_start + num_features : len(data[row_ix])]
        data_standardized.append(row_concat)

    return data_standardized


def cross_validate_per_subject(data_matrix, num_neighbors, f_standardize):

    data = standardize_features_per_subject(data_matrix, IX_HR, NUM_NUMERIC_FEATURES, f_standardize)

    subject_ids = get_subject_ids(data)

    for subj_id in subject_ids:
        test_data = get_rows_of_subject(subj_id, data)
        training_data = get_rows_of_other_subjects(subj_id, data)

        test_labels = get_labels(test_data)
        training_labels = get_labels(training_data)

        test_features = get_features(test_data)
        training_features = get_features(training_data)

        # Get nearest neighbors for every object in test set
        neighbors = knn.compute_nearest_neighbors(test_features, training_features, num_neighbors)

        predictions = []
        actuals = []
        for measurement in range(0, len(neighbors)):
            prediction = knn.majority_class(neighbors[measurement], training_labels)
            actual = test_labels[measurement]

            predictions.append(prediction)
            actuals.append(actual)

        c_ix = knn.c_index(actuals, predictions)
        print("(" + str(subj_id) + "," + str(c_ix) + ")")

'''
    for subj_id in subject_ids:
        misclassifications = 0
        classifications =  0

        test_data = get_rows_of_subject(subj_id, data)
        training_data = get_rows_of_other_subjects(subj_id, data)

        test_labels = get_labels(test_data)
        training_labels = get_labels(training_data)
        
        test_features = f_standardize(get_features(test_data))
        training_features = get_features(training_data)
        for i in range(0, len(training_features)):
            training_features[i] = f_standardize(training_features[i])


        #Get nearest neighbors for every object in test set
        neighbors = knn.compute_nearest_neighbors(test_features, training_features, num_neighbors)

        predictions = []
        actuals = []
        for measurement in range(0, len(neighbors)):

            prediction = knn.majority_class(neighbors[measurement], training_labels)
            actual = test_labels[measurement]

            predictions.append(prediction)
            actuals.append(actual)

        c_ix = knn.c_index(actuals, predictions)
        print("(" + str(subj_id) + "," + str(c_ix) + ")")
        #print("Classifications :" + str(classifications))
        #print("Misclassifications :" + str(misclassifications) + "\n")
   '''
def print_rows(data):
    for row in data:
        print(row)

def z_score_list(data):
    standardized = ndarray.tolist(stats.zscore(data.copy()))
    return standardized

data = open_and_parse(DATA_FILENAME)

k = 37

cross_validate_per_subject(data, k, z_score_list)
#print(get_rows_of_subject(3, data))