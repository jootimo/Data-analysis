import knn
import numpy as np

def random_X(num_rows, num_cols):
    X = np.random.randn(num_rows , num_cols)

    X_aslist = np.ndarray.tolist(X)

    if(len(X_aslist[0]) == 1):
        for x in X_aslist:
            x = x[0]

    return(X_aslist)

def random_Y(num_rows):
    Ypos = np.ones(num_rows / 2)
    Yneg = -1 * np.ones(num_rows / 2 + num_rows % 2)

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
    misclassifications = 0
    for test_ix, test_features in enumerate(features):
        training_features = [row for i, row in enumerate(features) if i != test_ix]
        training_labels = [row for i, row in enumerate(labels) if i != test_ix]

        prediction = f_predict([test_features], training_features, training_labels)
        predictions.append([prediction])
        
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



NUM_ROWS = [20, 50, 100, 500]
NUM_COLS = 1

NUM_NEIGHBORS = 3



print(random_X(20, NUM_COLS))
print(random_Y(20))