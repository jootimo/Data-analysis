import knn 
from scipy import stats
from numpy import mean

DATA_FILENAME = "data/Water_data.csv"

#Inputs
IX_MOD1         = 0
IX_MOD2         = 1
IX_MOD3         = 2

#Outputs
IX_C_TOTAL      = 0
IX_CD           = 1
IX_PB           = 2


# Open file with name filename and parse comma separated values a into 2-d list
#
# @param    filename
def open_and_parse(filename):
    data = []

    with open(filename, "r") as filestream:

        for line in iter(filestream.readline, ''):
            line = list(line.split(','))

            #First row is headers, skip it
            if line[0] == "Mod1":
                continue

            #Parse rows into lists
            for i in range(0, len(line)):
                for char in range(0, len(line[i])):

                    #Remove newline characters
                    if type(line[i]) == "str":
                        line[i].replace('\n', '')

                    #Cast to floats
                    line[i] = float(line[i])

            data.append(line)

    return data

# Compute C-index for list of predictions
#
# @param    true_values         True numeric target values for the data
# @param    predicted_values    Predicted target values
def c_index(true_values, predicted_values):
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

# Perform k-nearest-neighbors search with k-fold cross-validation
# and print c-index for each output attribute
#
# @param    inputs      Input attribute values in the data
# @param    outputs     Outputs resulted from inputs
# @param    num_folds   To how many folds the data will be divided
# @param    k           Number of neighbors to search
#
def regression_with_cross_validation_and_c_index(inputs, outputs, num_folds, k):
    
    c_total_predictions = []
    cd_predictions = []
    pb_predictions = []

    for fold in range(0, num_folds):

        #How many data objects in each fold
        len_fold = int(len(inputs) / num_folds)

        #Where is the test set placed in the data
        ix_test_first = fold * len_fold
        ix_test_one_past_last = ix_test_first + len_fold

        test_set = inputs[ix_test_first : ix_test_one_past_last]

        training_set = inputs[IX_MOD1 : ix_test_first] + inputs[ix_test_one_past_last :len(inputs)]
        training_set_outputs = (outputs[IX_C_TOTAL : ix_test_first] 
                            + outputs[ix_test_one_past_last :len(inputs)])

        #Get list of nearest neighbor indices for each object in the test set
        neighbors = knn.compute_nearest_neighbors(test_set, training_set, k)

        for i in range(0, len_fold):

            #Get the neighbors' output values 
            neighbors_c_totals = []
            neighbors_cds = []
            neighbors_pbs = []
            for n in neighbors:
                neighbors_c_totals.append(float(training_set_outputs[n][IX_C_TOTAL]))
                neighbors_cds.append(float(training_set_outputs[n][IX_CD]))
                neighbors_pbs.append(float(training_set_outputs[n][IX_PB]))

            #Mean value of the neigbors is the prediction
            estimate_c_total = mean(neighbors_c_totals)
            estimate_cd = mean(neighbors_cds)
            estimate_pb = mean(neighbors_pbs)
             
            #Store the predictions
            c_total_predictions.append(estimate_c_total)
            cd_predictions.append(estimate_cd)
            pb_predictions.append(estimate_pb)

    c_ix_c_total = c_index([row[IX_C_TOTAL] for row in outputs], c_total_predictions)
    c_ix_cd = c_index([row[IX_CD] for row in outputs], cd_predictions)
    c_ix_pb = c_index([row[IX_PB] for row in outputs], pb_predictions)

    print("C-index for c_totals: " + str(c_ix_c_total))
    print("C-index for cd: " + str(c_ix_cd))
    print("C-index for pb: " + str(c_ix_pb))



#Open file and read data into memory
data = open_and_parse(DATA_FILENAME)

data_standardized = stats.zscore(data)

#Partition data into inputs and outputs
inputs = []
outputs = []
for row in data_standardized:
    inputs.append(list(row[IX_MOD1:IX_MOD3 + 1]))
    outputs.append(list(row[len(inputs[0]) + IX_C_TOTAL : len(inputs[0]) + IX_PB + 1]))

#Data contains 4 measurements from each sample, 
#fold those measurements together
num_folds = int(len(data) / 4)

#Perform knn with k-fold cross-validation and print c-index for k = 1,2,...,5
for k in range(1, 6):
    print("C-index score for k = " + str(k) + " with " + str(num_folds) + 
        "-fold cross-validation:")
    regression_with_cross_validation_and_c_index(inputs, outputs, num_folds, k)
    print("")

for k in range(1, 6):
    print("C-index score for k = " + str(k) + " with " + 
        "leave-one-out cross-validation:")
    regression_with_cross_validation_and_c_index(inputs, outputs, len(data), k)
    print("")