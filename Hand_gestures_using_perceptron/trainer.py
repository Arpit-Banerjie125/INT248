import numpy
import tensorflow as tf

import glob
import cv2
from sklearn.utils import shuffle
import os
from tensorflow.keras.models import Sequential,model_from_json
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import to_categorical

seed = 7
numpy.random.seed(seed)


myFiveTrainImageFiles = glob.glob("Dataset/train/fiveFingerTrainDataset/*.png")
myFiveTrainImageFiles.sort()
myFiveTrainImages = [cv2.imread(img,0) for img in myFiveTrainImageFiles] #we pass zero to load greyscale image

for i in range(0,len(myFiveTrainImages)):
    myFiveTrainImages[i] = cv2.resize(myFiveTrainImages[i],(50,50))
tn1 = numpy.asarray(myFiveTrainImages)

myZeroTrainImageFiles = glob.glob("Dataset/train/zeroFingerTrainDataset/*.png")
myZeroTrainImageFiles.sort()
myZeroTrainImages = [cv2.imread(img,0) for img in myZeroTrainImageFiles]

for i in range(0,len(myZeroTrainImages)):
    myZeroTrainImages[i] = cv2.resize(myZeroTrainImages[i],(50,50))
tn2 = numpy.asarray(myZeroTrainImages)


finalTrainImages = []
finalTrainImages.extend(myFiveTrainImages)
finalTrainImages.extend(myZeroTrainImages)


myFiveTestImageFiles = glob.glob("Dataset/test/fiveFingerTestDataset/*.png")
myFiveTestImageFiles.sort()
myFiveTestImages = [cv2.imread(img,0) for img in myFiveTestImageFiles]


for i in range(0,len(myFiveTestImages)):
    myFiveTestImages[i] = cv2.resize(myFiveTestImages[i],(50,50))
ts1 = numpy.asarray(myFiveTestImages)

myZeroTestImageFiles = glob.glob("Dataset/test/zeroFingerTestDataset/*.png")
myZeroTestImageFiles .sort()
myZeroTestImages = [cv2.imread(img,0) for img in myZeroTestImageFiles]


for i in range(0,len(myZeroTestImages)):
    myZeroTestImages[i] = cv2.resize(myZeroTestImages[i],(50,50))
ts2 = numpy.asarray(myZeroTestImages)

finalTestImages = []
finalTestImages.extend(myFiveTestImages)
finalTestImages.extend(myZeroTestImages)



x_train = numpy.asarray(finalTrainImages)
x_test = numpy.asarray(finalTestImages)


y_myFiveTrainImages = numpy.empty([tn1.shape[0]])
y_myZeroTrainImages = numpy.empty([tn2.shape[0]])
y_myFiveTestImages = numpy.empty([ts1.shape[0]])
y_myZeroTestImages = numpy.empty([ts2.shape[0]])

for j in range(0,tn1.shape[0]):
    y_myFiveTrainImages[j] = 5

for j in range(0,ts1.shape[0]):
    y_myFiveTestImages[j] = 5

for j in range(0,tn2.shape[0]):
    y_myZeroTrainImages[j] = 0

for j in range(0,ts2.shape[0]):
    y_myZeroTestImages[j] = 0

y_train_temp = []
y_train_temp.extend(y_myFiveTrainImages)
y_train_temp.extend(y_myZeroTrainImages)
y_train = numpy.asarray(y_train_temp)

y_test_temp = []
y_test_temp.extend(y_myFiveTestImages)
y_test_temp.extend(y_myZeroTestImages)
y_test = numpy.asarray(y_test_temp)


print(x_train.shape)
#print(x_test.shape)

print(y_train.shape)
#print(y_test.shape)

x_train,y_train = shuffle(x_train,y_train)
x_test,y_test = shuffle(x_test,y_test)

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

x_train = x_train / 255
x_test = x_test / 255

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


num_classes = y_test.shape[1]
print("num_classes")
print(num_classes)




def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



model = baseline_model()

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=20, verbose=2)


scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model_json = model.to_json();
with open("perceptronModel.json","w") as jsonFile:
    jsonFile.write(model_json)
model.save_weights("perceptronModelWeights.h5")
print("Saved model to disk")











