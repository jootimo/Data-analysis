import knn

import numpy as np              # Random numbers
from functools import partial   # Function argument binding
import scipy.stats as stats     # Correlation
from operator import itemgetter # Sorting list of pairs on first item
import matplotlib               # Plotting
matplotlib.use('agg')
import matplotlib.pyplot as plt


def random_X(num_rows, num_cols):
    '''
    Generate random data matrix with values in ]0,1[

    @param  num_rows    number of rows to generate
    @param  num_cols    number of cols to generate

    @return 2-dimensional list
    '''

    X = np.random.randn(num_rows , num_cols)

    X_aslist = np.ndarray.tolist(X)

    return(X_aslist)

def random_Y(num_rows):
    '''
    Generate random label list with half its values being -1 and the other half 1 

    @param  num_rows    Number of rows to generate

    @return List of integers -1 and 1
    '''
    
    Ypos = np.ones(int(num_rows / 2))
    Yneg = -1 * np.ones(int(num_rows / 2 + num_rows % 2))

    Y = np.random.permutation(np.hstack((Ypos, Yneg)))
    return(np.ndarray.tolist(Y))

def loo_cv (features, labels, f_predict):
    '''
    Perform leave-one-out cross-validation, do classifications and report the accuracy

    @param  features    2d list of feature values
    @param  labels      Correct labels corresponding to rows in features
    @param  f_predict   Prediction function used for classifications

    @return Accuracy of the made predictions 
    '''

    predictions = []
    correct_predictions = 0
    for test_ix, test_features in enumerate(features):
        training_features = [row for i, row in enumerate(features) if i != test_ix]
        training_labels = [row for i, row in enumerate(labels) if i != test_ix]

        prediction = f_predict([test_features], training_features, training_labels)
        predictions.append(prediction)
        
        #See if we guessed correctly
        actual = labels[test_ix]
        if actual == prediction:
            correct_predictions += 1    
    

    num_rows = len(features)
    if num_rows != 0:
        accuracy = float(correct_predictions) / num_rows
    
    return(accuracy)



################################
######## EXCERCISE 1 ###########
# Variance of cross validation #
################################

num_rows = [20, 50, 100, 500]
num_cols = 1
num_neighbors = 3

f_predict = partial(knn.predict_classification, k = num_neighbors)

accuracies = []

#Try different sized data
for rows in num_rows:
    print("\nRows: " + str(rows))

    accuracies_of_this_round = []
    
    #Repeat test 100 times
    for i in range(100):
        X = random_X(rows, num_cols)
        Y = random_Y(rows)
        
        accuracies_of_this_round.append(loo_cv(X, Y, f_predict))
    
    mean_accuracy = np.mean(accuracies_of_this_round)
    var_accuracy = np.var(accuracies_of_this_round)
    print("Accuracy mean: " + str(mean_accuracy))
    print("Accuracy variance: " + str(var_accuracy))
    
    # How large a fraction (in %) of the leave-one-out runs resulted in performance higher than 0.60? What about 0.70?
    num_over_06 = 0
    num_over_07 = 0
    num_accuracies = len(accuracies_of_this_round)
    for acc in accuracies_of_this_round:
        if acc > 0.6:
            num_over_06 += 1
            if acc > 0.7:
                num_over_07 += 1

    percent_over_06 = (num_over_06 / num_accuracies) * 100
    percent_over_07 = (num_over_07 / num_accuracies) * 100

    print("Percentage of cross-validation runs with over 60% accuracy: " + str(percent_over_06))
    print("Percentage of cross-validation runs with over 70% accuracy: " + str(percent_over_07))

    #Plot a histogram
    plt.hist(accuracies_of_this_round, alpha=0.75, label=str(rows) + " rows")
    plt.legend(prop={'size': 10})
    plt.title("Prediction accuracies with 20, 50, 100 and 500 rows of random data")
    plt.xlabel("Prediction accuracy")
    plt.ylabel("Frequency")

    accuracies.append(accuracies_of_this_round)

plt.savefig("AccuracyHist.png")



################################
######## EXCERCISE 2 ###########
#      Feature selection       #
################################

def select_features(features, labels, num_selections):
    '''
    Get indices of the columns of a desired number of features that correlate most to the labels.
    Kendall tau coefficient is used.

    @param  features        2D list of the feature values where to select
    @param  labels          List of the corresponding labels
    @param  num_selections  How many features to select

    @return List of column indices
    '''

    #List of correlation-column index pairs
    correlations = []
    for ix_col in range(len(features[0])):
        column = [row[ix_col] for row in features]

        tau, p_val = stats.kendalltau(column, labels)
        correlations.append([tau, ix_col])

    #Get absolute values
    for corr in correlations:
        corr[0] = abs(corr[0])

    #Sort the correlation-index pairs on the correlation value in descending order
    correlations.sort(key = itemgetter(0), reverse=True)
    selections = correlations[0 : num_selections]

    #Select only the requested amount of columns
    selected_indices = []
    for ix_row in range(len(features)):
        row = []
        for sel in selections:
            selected_indices.append(sel[1])
        
    return(selected_indices)

def loo_cv_with_feat_selection (features, labels, num_features, f_predict):
    '''
    Select the best features from a dataset, perform leave-one-out cross-validation,
    do classifications and report the accuracy.

    @param  features    2d list of feature values
    @param  labels      Correct labels corresponding to rows in features
    @param  f_predict   Prediction function used for classifications

    @return Accuracy of the made predictions 
    '''

    predictions = []
    correct_predictions = 0
    for test_ix, test_features in enumerate(features):
    
        training_features = [row for i, row in enumerate(features) if i != test_ix]
        training_labels = [row for i, row in enumerate(labels) if i != test_ix]
        selected_feature_indices = select_features(training_features, training_labels, num_features)

        #Create test and training lists that only contain the selected columns
        training_features_sel = []
        test_features_sel = []
        for feat_row in training_features:
            row = []
            for ix in selected_feature_indices:
                row.append(feat_row[ix])
            training_features_sel.append(row)

        for ix in selected_feature_indices:
            test_features_sel.append(test_features[ix])

        #Predict label        
        prediction = f_predict([test_features_sel], training_features_sel, training_labels)
        predictions.append(prediction)
        
        #See if we guessed correctly
        actual = labels[test_ix]
        if actual == prediction:
            correct_predictions += 1    
    
    #Compute accuracy of predictions
    num_rows = len(features)
    if num_rows != 0:
        accuracy = float(correct_predictions) / num_rows
    
    return(accuracy)



num_rows = 30
num_cols = 100

accuracies_wrong = []
accuracies_right = []

plt.clf() #Reset plot

for i in range(100):

    X = random_X(num_rows, num_cols)
    Y = random_Y(num_rows)
    f_predict = partial(knn.predict_classification, k = 3)

    #How feature selection should be done. New selections on each round of cross-validation
    accuracies_right.append(loo_cv_with_feat_selection(X, Y, 10, f_predict))

    # The wrong way to do feature selection: as a preprocessing step.
    X_selected_indices = select_features(X, Y, 10)
    X_selected_features = []
    for feat_row in X:
        row = []
        for ix in X_selected_indices:
            row.append(feat_row[ix])
        X_selected_features.append(row)

    accuracies_wrong.append(loo_cv(X_selected_features, Y, f_predict))


print("Mean accuracy of the wrong way: " + str(np.mean(accuracies_wrong)))
print("\nMean accuracy of the right way: " + str(np.mean(accuracies_right)))


plt.hist(accuracies_wrong, alpha=0.75, label="Wrong way")
plt.hist(accuracies_right, alpha=0.75, label="Right way")

plt.legend(prop={'size': 10})
plt.title("Prediction accuracies with feature selections done wrong and right")
plt.xlabel("Prediction accuracy")
plt.ylabel("Frequency")
plt.savefig("FeatureSelection.png")
