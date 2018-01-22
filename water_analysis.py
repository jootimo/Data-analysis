import knn 
from scipy import stats
from numpy import mean

DATA_FILENAME = "data/Water_data.csv"

IX_MOD1         = 0
IX_MOD2         = 1
IX_MOD3         = 2
IX_C_TOTAL      = 3
IX_CD           = 4
IX_PB           = 5


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

# Perform k-nearest-neighbors search with k-fold crossvalidation
#
# @param    inputs      Input attribute values in the data
# @param    outputs     Outputs resulted from inputs
# @param    num_folds   To how many folds the data will be divided
# @param    k           Number of neighbors to search
#
# @return   Average C-index value
def regression_with_cross_validation_and_c_index(inputs, outputs, num_folds, k):
    
    sum_c_index = 0
    num_c_index_computations = 0
    
    for fold in range(0, num_folds):

        #How many data objects in each fold
        len_fold = int(len(inputs) / num_folds)

        #Where is the test set placed in the data
        ix_test_first = fold * len_fold
        ix_test_last = ix_test_first + len_fold

        test_set = inputs[ix_test_first : ix_test_last]
        test_set_outputs = outputs[ix_test_first : ix_test_last]

        training_set = inputs[0 : ix_test_first] + inputs[ix_test_last :len(inputs)]
        training_set_outputs = outputs[0 : ix_test_first] + outputs[ix_test_last :len(inputs)]        

        #Get list of nearest neighbors for each object in the test set
        neighbors = knn.compute_nearest_neighbors(test_set, training_set, k)

        for i in range(0, len_fold):
            
            #Get the indices of the neighbors of the current test object
            ixs_neighbors = []
            for j in range(0, len(neighbors[i])):
                ixs_neighbors.append(neighbors[i][j][0])

            #Get the neighbors' output values and calculate their mean 
            neighbors_c_totals = []
            neighbors_cds = []
            neighbors_pbs = []
            
            for l in ixs_neighbors:
                neighbors_c_totals.append(float(training_set_outputs[l][0]))
                neighbors_cds.append(float(training_set_outputs[l][1]))
                neighbors_pbs.append(float(training_set_outputs[l][2]))

            estimate_c_total = mean(neighbors_c_totals)
            estimate_cd = mean(neighbors_cds)
            estimate_pb = mean(neighbors_pbs)


            #Combine the test set predictions with the training set for c-index
            #computation
            outputs_combined = (training_set_outputs[0 : ix_test_first] 
                            + [[estimate_c_total, estimate_cd, estimate_pb]]
                            + training_set_outputs[ix_test_first : ] )

            #Compute c-index by comparing the predicted outputs and the actual outputs
            sum_c_index += c_index(outputs_combined, outputs)
            num_c_index_computations += 1
            
    return(sum_c_index / num_c_index_computations)



#Open file and read data into memory
data = open_and_parse(DATA_FILENAME)

data_standardized = stats.zscore(data)

#Partition data into inputs and outputs
inputs = []
outputs = []
for row in data_standardized:
    inputs.append(list(row[IX_MOD1:IX_MOD3 + 1]))
    outputs.append(list(row[IX_MOD3:IX_PB + 1]))

#Data contains 4 measurements from each sample, 
#fold those measurements together
num_folds = int(len(data) / 4)

#Perform knn and print c-index for k = 1,2,...,5
for k in range(1, 6):
    print("C-index score for k = " + str(k) + ":")
    print(regression_with_cross_validation_and_c_index(inputs, outputs, num_folds, k))
