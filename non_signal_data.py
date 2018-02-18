import knn

import numpy as np              # Random numbers
from functools import partial   # Function argument binding
import scipy.stats as stats     # Correlation
from operator import itemgetter
import matplotlib               # Plotting
matplotlib.use('agg')
import matplotlib.pyplot as plt


def random_X(num_rows, num_cols):
    X = np.random.randn(num_rows , num_cols)

    X_aslist = np.ndarray.tolist(X)

    return(X_aslist)

def random_Y(num_rows):
    Ypos = np.ones(int(num_rows / 2))
    Yneg = -1 * np.ones(int(num_rows / 2 + num_rows % 2))

    Y = np.random.permutation(np.hstack((Ypos, Yneg)))
    return(np.ndarray.tolist(Y))

def loo_cv (features, labels, f_predict):
    '''
    Perform leave-one-out cross-validation, do classifications and report the c-index

    @param  features    2d list of feature values
    @param  labels      Correct labels corresponding to rows in features
    @param  f_predict   Prediction function used for classifications

    @return Concordance-index for the made predictions 
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
for rows in num_rows:
    print("\nRows: " + str(rows))

    accuracies_of_this_round = []
    for i in range(100):
        X = random_X(rows, num_cols)
        Y = random_Y(rows)
        
        accuracies_of_this_round.append(loo_cv(X, Y, f_predict))
    
    mean_accuracy = np.mean(accuracies_of_this_round)
    var_accuracy = np.var(accuracies_of_this_round)
    print("Accuracy mean: " + str(mean_accuracy))
    print("Accuracy variance: " + str(var_accuracy))
    
    # How large fraction (in %) of the leave-one-out runs resulted in performance higher than 0.60? What about 0.70?
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
    selected_features = []
    for ix_row in range(len(features)):
        row = []
        for sel in selections:
            ix_col = sel[1]
            row.append(features[ix_row][ix_col])
        
        selected_features.append(row)

    return(selected_features)

def loo_cv_with_feat_selection (features, labels, num_features, f_predict):
    '''
    Perform leave-one-out cross-validation, do classifications and report the c-index

    @param  features    2d list of feature values
    @param  labels      Correct labels corresponding to rows in features
    @param  f_predict   Prediction function used for classifications

    @return Concordance-index for the made predictions 
    '''

    predictions = []
    correct_predictions = 0
    for test_ix, test_features in enumerate(features):
        test_features_sel = select_features([test_features], [labels[test_ix]], num_features)
    
        training_features = [row for i, row in enumerate(features) if i != test_ix]
        training_labels = [row for i, row in enumerate(labels) if i != test_ix]
        training_features_sel = select_features(training_features, training_labels, num_features)

        prediction = f_predict(test_features_sel, training_features_sel, training_labels)
        predictions.append(prediction)
        
        #See if we guessed correctly
        actual = labels[test_ix]
        if actual == prediction:
            correct_predictions += 1    
    
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

    X_selected = select_features(X, Y, 10)
    f_predict = partial(knn.predict_classification, k = 3)

    accuracies_wrong.append(loo_cv(X_selected, Y, f_predict))

    accuracies_right.append(loo_cv_with_feat_selection(X, Y, 10, f_predict))

print("Mean accuracy of the wrong way: " + str(np.mean(accuracies_wrong)))
print("\nMean accuracy of the right way: " + str(np.mean(accuracies_right)))


plt.hist(accuracies_wrong, alpha=0.75, label="Wrong way")
plt.hist(accuracies_right, alpha=0.75, label="Right way")

plt.legend(prop={'size': 10})
plt.title("Prediction accuracies with feature selections done wrong and right")
plt.xlabel("Prediction accuracy")
plt.ylabel("Frequency")
plt.savefig("FeatureSelection.png")

'''

Rows: 20
Accuracy mean: 0.4425
Accuracy variance: 0.02311875
Percentage of cross-validation runs with over 60% accuracy: 10.0
Percentage of cross-validation runs with over 70% accuracy: 5.0

Rows: 50
Accuracy mean: 0.49420000000000003
Accuracy variance: 0.009058360000000001
Percentage of cross-validation runs with over 60% accuracy: 12.0
Percentage of cross-validation runs with over 70% accuracy: 1.0

Rows: 100
Accuracy mean: 0.48869999999999997
Accuracy variance: 0.00508131
Percentage of cross-validation runs with over 60% accuracy: 7.000000000000001
Percentage of cross-validation runs with over 70% accuracy: 0.0

Rows: 500
Accuracy mean: 0.49838
Accuracy variance: 0.0010084556000000003
Percentage of cross-validation runs with over 60% accuracy: 0.0
Percentage of cross-validation runs with over 70% accuracy: 0.0
Mean accuracy of the wrong way: 0.7463333333333333
Mean accuracy of the right way: 0.47733333333333333



Rows: 20
Accuracy mean: 0.46649999999999997
Accuracy variance: 0.022902750000000003
Percentage of cross-validation runs with over 60% accuracy: 14.000000000000002
Percentage of cross-validation runs with over 70% accuracy: 2.0

Rows: 50
Accuracy mean: 0.4854
Accuracy variance: 0.010326840000000002
Percentage of cross-validation runs with over 60% accuracy: 14.000000000000002
Percentage of cross-validation runs with over 70% accuracy: 1.0

Rows: 100
Accuracy mean: 0.5054000000000001
Accuracy variance: 0.00440484
Percentage of cross-validation runs with over 60% accuracy: 8.0
Percentage of cross-validation runs with over 70% accuracy: 0.0

Rows: 500
Accuracy mean: 0.49648
Accuracy variance: 0.0010772896000000006
Percentage of cross-validation runs with over 60% accuracy: 0.0
Percentage of cross-validation runs with over 70% accuracy: 0.0

Mean accuracy of the wrong way: 0.7476666666666668
Mean accuracy of the right way: 0.4676666666666667
'''


'''
accuracies_flattened = []
for ix, accuracy_list in enumerate(accuracies):
    for ix_acc, acc in enumerate(accuracy_list):
        accuracies_flattened.append(acc)

print("\nOverall mean accuracy: " + str(np.mean(accuracies_flattened)))

# How large fraction (in %) of the leave-one-out runs resulted in performance higher than 0.60? What about 0.70?
num_over_06 = 0
num_over_07 = 0
num_accuracies = len(accuracies_flattened)
for acc in accuracies_flattened:
    if acc > 0.6:
        num_over_06 += 1
        if acc > 0.7:
            num_over_07 += 1

percent_over_06 = (num_over_06 / num_accuracies) * 100
percent_over_07 = (num_over_07 / num_accuracies) * 100

print("Overall percentage of cross-validation runs with over 60% accuracy: " + str(percent_over_06))
print("Overall percentage of cross-validation runs with over 70% accuracy: " + str(percent_over_07))

#Plot a histogram
plt.hist(accuracies_flattened, 20)
plt.title("Accuracies from " + str(num_accuracies) + " cross-validation runs")
plt.xlabel("Prediction accuracy")
plt.ylabel("Frequency")
plt.savefig("AccuracyHist.png")

#fig = plt.gcf()

#plot_url = py.plot_mpl(fig, filename='Accuracy_hist')
'''