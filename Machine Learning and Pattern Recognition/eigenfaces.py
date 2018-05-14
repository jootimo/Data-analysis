from sklearn.decomposition import PCA
import read_yale

a = read_yale.load_images_croppedyale() #tuple: pics, fnames, lnames
#b = read_yale.get_croppedyale_as_df()
pics = a[0]
suffixes = [ a[2][0] ]
c = read_yale.images_to_array(pics, suffixes)
print(type(c))
print(type(c[0][0]))
print(len(c[0]))
print(len(c[0][0]))


pic_arrays = c[0]

'''
picno = 0
for pic in pic_arrays:
    pixno = 0
    pixsum = 0
    for pixel in pic:
        pixsum = pixsum + pixel 
        pixno = pixno + 1

    print("picno " + str(picno) + " avg = " + str(pixsum / float(pixno)))
    picno = picno + 1
'''

pca = PCA(n_components = 10)
pca.fit(pic_arrays)

print(pca.explained_variance_ratio_)  

'''names = c[1]
prev_name = ""
for name in names:
    if prev_name != name:
        print(name)
        prev_name = name
'''