import knn

DATA_FILENAME = "iris.data"

#Classes as binary values
IRIS_SETOSA =       0b001
IRIS_VERSICOLOUR =  0b010
IRIS_VIRGINICA =    0b100

#Indices of feature names
IX_SEPAL_LENGTH =   0
IX_SEPAL_WIDTH =    1
IX_PETAL_LENGTH =   2
IX_PETAL_WIDTH =    3
IX_IRIS_CLASS =     4

NUM_FEATURES =      5

# Open file with name filename and parse comma separated values a into 2-d list
#
# @param    filename
# @param    features    A 2-d list where data values will be stored
def open_and_parse(filename, features):
    with open(filename, "r") as filestream:

        for line in iter(filestream.readline, ''):
            current_line = line.split(',')

            row = [0] * NUM_FEATURES

            row[IX_SEPAL_LENGTH]  = float(current_line[0])
            row[IX_SEPAL_WIDTH]  = float(current_line[1])
            row[IX_PETAL_LENGTH]  = float(current_line[2])
            row[IX_PETAL_WIDTH]  = float(current_line[3])
            
            current_class = current_line[4]
            if current_class == "Iris-setosa\n" or current_class == "Iris-setosa":
                row[IX_IRIS_CLASS]  = IRIS_SETOSA
            
            if current_class == "Iris-versicolor\n" or current_class == "Iris-versicolor":
                row[IX_IRIS_CLASS]  = IRIS_VERSICOLOUR
                
            if current_class == "Iris-virginica\n" or current_class == "Iris-virginica":
                row[IX_IRIS_CLASS]  = IRIS_VIRGINICA


            features.append(row)

#Open file and read data into memory
features = list()

open_and_parse(DATA_FILENAME, features)

ks_to_test = [1, 2, 3, 5, 8, 13]
for k in ks_to_test:
    print("k = " + str(k))
    print(knn.classification_with_cross_validation(features, IX_IRIS_CLASS, k))
