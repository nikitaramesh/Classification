import cv2
import argparse
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
arguments = vars(ap.parse_args())
imagePaths = list(paths.list_images(arguments["dataset"]))
data = []
labels = []

def convert_image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()

for (i, imagePath) in enumerate(imagePaths):
    # input format: /{data_path}/{class}.{image_no}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # construct a feature vector
    features = convert_image_to_feature_vector(image)
    data.append(features)
    labels.append(label)
    if i > 0 and i % 100 == 0:
        print("processed {}/{}".format(i, len(imagePaths)))

# encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 6)
 
# partition the data into training and testing data, using 75%
# of the data for training and the remaining 25% for testing
print("constructing training/testing split...")
(train_Data, test_Data, train_Labels, test_Labels) = train_test_split(data, labels, test_size=0.25, random_state=42)

# architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072, kernel_initializer="uniform", activation="relu"))
model.add(Dense(384, kernel_initializer="uniform", activation="relu"))
model.add(Dense(6))
model.add(Activation("softmax"))

# train the model using SGD
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(train_Data, train_Labels, epochs=50, batch_size=128, verbose=1)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_Data, test_Labels, batch_size=128, verbose=1)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))