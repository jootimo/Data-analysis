import knn 
from scipy import stats
from numpy import mean, std

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
        features.append(row[IX_HR : IX_RMSORB])
        
    return features

def get_labels(data):
    labels = []
    for row in data:
        labels.append(row[IX_LABEL])

    return labels

def cross_validate_per_subject(data, num_neighbors, f_standardize, f_print):
    subject_ids = get_subject_ids(data)
    for id in subject_ids:
        test_data = get_rows_of_subject(id, data)
        training_data = get_rows_of_other_subjects(id, data)
        
        test_labels = get_labels(test_data)
        training_labels = get_labels(training_data)
        
        test_features = f_standardize(get_features(test_data))
        training_features = f_standardize(get_features(training_data))

        neighbors = knn.compute_nearest_neighbors(test_features, training_features, num_neighbors)

        prediction = knn.majority_class(neighbors, training_labels)

        print("predicted : " + str(prediction))
        print("actual : " + str(test_labels))
        



    
def print_rows(data):
    for row in data:
        print(row)


data = open_and_parse(DATA_FILENAME)
z_score = lambda x: (x - mean(x)) / std(x)
k = 37

cross_validate_per_subject(data, k, z_score, print_rows)
#print(get_rows_of_subject(3, data))