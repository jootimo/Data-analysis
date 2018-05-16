from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np

import read_yale

def save_png(pixels, width, height):
    plt.imshow(pixels.reshape(height, width), cmap=plt.cm.bone)
    #plt.savefig("img.png")
    plt.show()

def distance_betw_imgs(img1, img2):
    d = 0
    if(len(img1) != len(img2)):
        return -1

    for px in range(len(img1)):
        d = d + abs(img1[px] - img2[px])

    return d

def ix_closest(img, other_imgs):
    min_d = 9999999
    ix = -1
    for i in range(len(other_imgs)):
        d = distance_betw_imgs(img, other_imgs[i])
        print("d=" + str(d))
        if d < min_d:
            min_d = d
            ix = i 

    print("ix = " + str(ix))
    return ix

def eucl_distance(x, y):
    return np.sqrt(np.sum(x - y) ** 2)

def nearest_eucl(target, others):
    ix = -1
    min_d = 999999
    for i in range(len(others)):
        d = eucl_distance(target, others[i])
        if d < min_d:
            min_d = d
            ix = i
        
    #print("min_d = " + str(min_d))
    return ix

a = read_yale.load_images_croppedyale() #tuple: pics, fnames, lnames
#b = read_yale.get_croppedyale_as_df()
pics = a[0]

#print(len(item[0]))
#print(item[0])
#print(len(item[1]))
#print(item[1])
suffixes = a[2]
#suffixes = suffixes[0:13] # to speed things up
test_suffix_ix = 1
training_suffixes = suffixes
test_suffix = training_suffixes.pop(test_suffix_ix)

training = read_yale.images_to_array(pics, training_suffixes)
test = read_yale.images_to_array(pics, [test_suffix])

test_pixels = test[0]
test_labels = test[1]
test_label_picname_pairs = test[2]
test_img_resolutions = test[3]
test_n_pics = len(test[0])

training_pixels = training[0]
training_labels = training[1]
training_label_picname_pairs = training[2]
training_img_resolutions = training[3]
training_n_pics = len(training[0])


#Create PCA space from training set 
pca = PCA(n_components = 100)
pca.fit(training_pixels)
eigenfaces = pca.components_

IMG_WIDTH = 168
IMG_HEIGHT = 192

print(training_pixels[0])
print(test_pixels[0])

training_trf = pca.transform(training_pixels)
test_trf = pca.transform(test_pixels)

correct = 0
incorrect = 0
for ix, item in enumerate(test_trf):
    closest = nearest_eucl(item, training_trf)
    if test_labels[ix] == training_labels[closest]:
        correct = correct + 1
    else:
        incorrect = incorrect + 1
    #print("label test : " + test_labels[ix])
    #print("label closest : " + training_labels[closest])
    #save_png(test_pixels[ix], IMG_WIDTH, IMG_HEIGHT)
    #save_png(training_pixels[closest], IMG_WIDTH, IMG_HEIGHT)

print("Correct " + str(correct) + "/" + str(correct + incorrect))

'''
print(len(eigenfaces))
for pic in test_pixels:
    closest = ix_closest(pic, eigenfaces)
    save_png(pic, IMG_WIDTH, IMG_HEIGHT)
    save_png(eigenfaces[closest], IMG_WIDTH, IMG_HEIGHT)
'''
#Transform a test pic
#transformed = pca.transform(test_pixels)
#print(len(test_pixels[0]))
#print(len(transformed[0]))
#for pic in transformed:
    #save_png(pic, IMG_WIDTH, IMG_HEIGHT)


'''
print(len(c[0]))
print(str(len(c[0][0])) + " : ")
print(c[0][0])
print(str(len(c[1][0])) + " : ")
print(c[1][0])
print(str(len(c[2][0])) + " : ")
print(c[2][0])
print("int : ")
print(c[3][0])
'''



#pca = PCA(n_components = 150)
#pca.fit(pic_arrays)
#test_pic = pca.transform([pic_arrays[20]])

#print(pca.explained_variance_ratio_)  


#save_png(pca.components_[24], IMG_WIDTH, IMG_HEIGHT)

'''names = c[1]
prev_name = ""
for name in names:
    if prev_name != name:
        print(name)
        prev_name = name
'''
