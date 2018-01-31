import knn                      # K-nearest-neighbors utilities
from functools import partial   # Function argument binding
from numpy import ndarray       # array -> list conversion
from scipy import stats         # z-score

FILE_INPUTS = "data/permeability/input.csv"
FILE_OUTPUTS = "data/permeability/output.csv"
FILE_COORDINATES = "data/permeability/coordinates.csv"

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


                    line[i] = float(line[i])

            data.append(line)

    return data


def remove_nearby_points(test_set, training_set, delta):

	print(test_set)
	print(training_set)
	distances = knn.compute_distances([test_set], [training_set])
	

	result = []
	
	for i in range(0, len(test_set)):
		if(distances[i] > delta):
			result.append(training_set[i])
			


def spatial_loo_cv(inputs, delta, f_predict):
	
	for ix_test in range(0, len(inputs)):
		test_set = inputs[ix_test]
		
		training_set = remove_nearby_points(test_set, inputs[:ix_test] + inputs[ix_test + 1:], delta)
		
		print(test_set)




INPUTS = open_and_parse(FILE_INPUTS) # 95 rows
OUTPUTS = open_and_parse(FILE_OUTPUTS)
COORDINATES = open_and_parse(FILE_COORDINATES)

INPUTS_STD = stats.zscore(INPUTS)

DS_TO_TEST = range(0,200)[0::10]
for d in DS_TO_TEST:
	spatial_loo_cv(INPUTS, 10, d)
