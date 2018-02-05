import knn                      # K-nearest-neighbors utilities
from functools import partial   # Function argument binding
from numpy import mean          # mean
from scipy import stats         # z-score

import matplotlib 
matplotlib.use('pdf')
import matplotlib.pyplot as plt #Plotting results

def open_and_parse(filename):
    ''' Open file with name filename and parse comma separated values a into 2-d list
        
        @param    filename
    '''
    data = []

    with open(filename, "r") as filestream:

        for line in iter(filestream.readline, ''):
            line = list(line.split(','))

            #Parse rows into lists
            for i in range(0, len(line)):

                #Remove newline characters
                line[i].replace('\n', '')

                line[i] = float(line[i])

            data.append(line)

    return data


def get_nearby_pnt_indices(pnts, ix_pnt, coordinates, delta):
    ''' Get indices of points in pnts that are close to the specified point

        @param    pnts        List of data points
        @param    ix_pnt      Index of the point in pnts to compare to
        @param    coordinates 2d Coordinates of the pnts
        @param    delta       Max distance of what is considered 'nearby'
    '''
    ixs = []
    for ix_other_pnt, item in enumerate(pnts):
        dist = knn.distance(coordinates[ix_pnt], coordinates[ix_other_pnt])

        if dist < delta:
            ixs.append(ix_other_pnt)

    return(ixs)



def spatial_loo_cv(inputs, outputs, coordinates, delta, f_predict):
    '''
    Perform spatial leave-one-out cross-validation for data and call a
    prediction function.

    @param  inputs      2d list of the input data
    @param  outputs     1d list of the corresponding outputs
    @param  coordinates List of 2d coordinates of where the measurements were made
    @param  delta       Size of the dead zone. Same unit as in coordinates
    @param  f_predict   Function used to make the predictions

    @return C-index value of the predictions
    '''

    predictions = []
    for test_ix, test_inputs in enumerate(inputs):
        nearby_pnt_ixs = get_nearby_pnt_indices(inputs, test_ix, coordinates, delta)

        #Remove data of nearby points
        training_outputs = []
        training_inputs = []
        for ix, item in enumerate(inputs): 
            if (ix != test_ix) and (ix not in nearby_pnt_ixs):
                training_inputs.append(inputs[ix])
                training_outputs.append(outputs[ix])

        #Make prediction for test set
        prediction = f_predict([test_inputs], training_inputs, training_outputs)
        predictions.append(prediction)

    return(knn.c_index(outputs, predictions))



######### Actual script ##########
FILE_INPUTS = "data/permeability/input.csv"
FILE_OUTPUTS = "data/permeability/output.csv"
FILE_COORDINATES = "data/permeability/coordinates.csv"

inputs = open_and_parse(FILE_INPUTS)
outputs = open_and_parse(FILE_OUTPUTS)
coordinates = open_and_parse(FILE_COORDINATES)

#Standardize inputs with z-score
inputs_std = stats.zscore(inputs)

#Test with delta values 0, 10, 20, ..., 200
ds_to_test = range(0, 201)[0::10]


'''
#Test nearest neighbors regression for different values of k
#and delta to find out the best values for k
ks_to_test = [1, 3, 5, 7, 9]
for d in ds_to_test:

    print("delta = " + str(d))

    c_ixs = []
    for num_neighbors in ks_to_test:    
        prediction_func = partial(knn.predict_regression, k=num_neighbors)

        c_ix = spatial_loo_cv(inputs_std, outputs, coordinates, d, prediction_func)
        print("k = " + str(num_neighbors) + ", C-index = " + str(c_ix))
        c_ixs.append(c_ix)
    
    print("Average C-index for delta = " + str(d) + ": " + str(mean(c_ixs)))
'''

#Best c-index values were achived when using k = 9 for delta < 100 and
#k = 7 for 100 <= delta <= 200
#
#Do predictions again, but this time only with the best k. Plot the 
#resulting c-indices.
c_ixs = []
for d in ds_to_test:
    best_k = int()

    if d < 100:
        best_k = 9
    else: 
        best_k = 7
    
    prediction_func = partial(knn.predict_regression, k=best_k)

    c_ix = spatial_loo_cv(inputs_std, outputs, coordinates, d, prediction_func)
    print("k = " + str(best_k) + ", delta = " + str(d) + ", C-index = " + str(c_ix))
    c_ixs.append(c_ix)


print("Average C-index for the best values of k: " + str(mean(c_ixs)))

plt.plot(ds_to_test, c_ixs)
plt.xlabel('Delta')
plt.ylabel('C-index')
plt.title('C-indices with the best values of k for delta = 0, 10, ..., 200')
plt.xticks(range(0, 200, 20))
plt.show()
plt.savefig("c_idx_plot.png")


'''
best k: 7: 11 hits, 9: 10 hits

delta = 0
k = 1, C-index = 0.6733448453023747
k = 3, C-index = 0.7084659169654319
k = 5, C-index = 0.717279082887577
k = 7, C-index = 0.7206332796232549
k = 9, C-index = 0.7224188714163922
Average C-index for delta = 0: 0.7084283992390061
delta = 10
k = 1, C-index = 0.6744175302976639
k = 3, C-index = 0.713814292853853
k = 5, C-index = 0.718894934966941
k = 7, C-index = 0.7194374020934331
k = 9, C-index = 0.7209458106838725
Average C-index for delta = 10: 0.7095019941791527
delta = 20
k = 1, C-index = 0.6632315081706048
k = 3, C-index = 0.6979462544569799
k = 5, C-index = 0.7065896806653517
k = 7, C-index = 0.7098042358633067
k = 9, C-index = 0.7116115263415037
Average C-index for delta = 20: 0.6978366410995493
delta = 30
k = 1, C-index = 0.6601579944255378
k = 3, C-index = 0.7002494648824281
k = 5, C-index = 0.7079685971030155
k = 7, C-index = 0.7106193364682101
k = 9, C-index = 0.7115747785684188
Average C-index for delta = 30: 0.698114034289522
delta = 40
k = 1, C-index = 0.6607323096220369
k = 3, C-index = 0.7010190682444644
k = 5, C-index = 0.7030888428161254
k = 7, C-index = 0.7048537858605769
k = 9, C-index = 0.7053528556169496
Average C-index for delta = 40: 0.6950093724320306
delta = 50
k = 1, C-index = 0.6578691331305323
k = 3, C-index = 0.6969946621234753
k = 5, C-index = 0.6979147563657643
k = 7, C-index = 0.7001644200361458
k = 9, C-index = 0.7043442167404655
Average C-index for delta = 50: 0.6914574376792766
delta = 60
k = 1, C-index = 0.6488498297003201
k = 3, C-index = 0.6888055083861918
k = 5, C-index = 0.6941461847412047
k = 7, C-index = 0.6972662456655127
k = 9, C-index = 0.7008605278520121
Average C-index for delta = 60: 0.6859856592690484
delta = 70
k = 1, C-index = 0.6449762644383751
k = 3, C-index = 0.6856584990949548
k = 5, C-index = 0.6927392700002379
k = 7, C-index = 0.6954950030028181
k = 9, C-index = 0.6992859732700198
Average C-index for delta = 70: 0.6836310019612811
delta = 80
k = 1, C-index = 0.6422856274909741
k = 3, C-index = 0.682282253695426
k = 5, C-index = 0.6871613080247337
k = 7, C-index = 0.6914698969242464
k = 9, C-index = 0.6943880200859828
Average C-index for delta = 80: 0.6795174212442726
delta = 90
k = 1, C-index = 0.6404188406182585
k = 3, C-index = 0.6821919591672745
k = 5, C-index = 0.6860504753411943
k = 7, C-index = 0.6901427773476927
k = 9, C-index = 0.69102157409261
Average C-index for delta = 90: 0.677965125313406
delta = 100
k = 1, C-index = 0.6416570655818258
k = 3, C-index = 0.680752146419927
k = 5, C-index = 0.6849361428697421
k = 7, C-index = 0.688658517293852
k = 9, C-index = 0.6878542660314785
Average C-index for delta = 100: 0.676771627639365
delta = 110
k = 1, C-index = 0.6088336046835562
k = 3, C-index = 0.6398459253369246
k = 5, C-index = 0.6474481646412228
k = 7, C-index = 0.6546195800534488
k = 9, C-index = 0.6492953527016263
Average C-index for delta = 110: 0.6400085254833557
delta = 120
k = 1, C-index = 0.5893499353939151
k = 3, C-index = 0.621438790809277
k = 5, C-index = 0.6281251356167816
k = 7, C-index = 0.6341685693846952
k = 9, C-index = 0.6299870227864192
Average C-index for delta = 120: 0.6206138907982177
delta = 130
k = 1, C-index = 0.5855253671627499
k = 3, C-index = 0.613806103350137
k = 5, C-index = 0.6185784141481027
k = 7, C-index = 0.6210527642024893
k = 9, C-index = 0.620620540395252
Average C-index for delta = 130: 0.6119166378517462
delta = 140
k = 1, C-index = 0.5857206553282871
k = 3, C-index = 0.61117916254275
k = 5, C-index = 0.618185037986698
k = 7, C-index = 0.6202149149761524
k = 9, C-index = 0.6188233993020023
Average C-index for delta = 140: 0.610824634027178
delta = 150
k = 1, C-index = 0.5844134845428367
k = 3, C-index = 0.6078662633044437
k = 5, C-index = 0.6150481780804083
k = 7, C-index = 0.6166027838712974
k = 9, C-index = 0.6144311654713724
Average C-index for delta = 150: 0.6076723750540717
delta = 160
k = 1, C-index = 0.5828403298760095
k = 3, C-index = 0.6082617393385961
k = 5, C-index = 0.6155192495334783
k = 7, C-index = 0.6171172526944867
k = 9, C-index = 0.6147720448140843
Average C-index for delta = 160: 0.6077021232513309
delta = 170
k = 1, C-index = 0.5832018579674072
k = 3, C-index = 0.6085588713323973
k = 5, C-index = 0.6151454721843856
k = 7, C-index = 0.6169510127686262
k = 9, C-index = 0.6144675632656661
Average C-index for delta = 170: 0.6076649555036965
delta = 180
k = 1, C-index = 0.5836155328987064
k = 3, C-index = 0.6076258278748308
k = 5, C-index = 0.6143408709432209
k = 7, C-index = 0.6166699797992242
k = 9, C-index = 0.6132356379203421
Average C-index for delta = 180: 0.6070975698872649
delta = 190
k = 1, C-index = 0.5840096090176935
k = 3, C-index = 0.6060509233140472
k = 5, C-index = 0.6134092274008195
k = 7, C-index = 0.6160141195443556
k = 9, C-index = 0.6126998203908843
Average C-index for delta = 190: 0.60643673993356
delta = 200
k = 1, C-index = 0.5842776927718181
k = 3, C-index = 0.6055483537697616
k = 5, C-index = 0.6124040883122482
k = 7, C-index = 0.615457653266212  
k = 9, C-index = 0.6114192479935716
Average C-index for delta = 200: 0.6058214072227223
'''