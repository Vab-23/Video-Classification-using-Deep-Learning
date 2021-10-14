#Import relavent Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import matplotlib

#Change the drive to the script location
os.chdir("E:\\Sem 5\\PROJECTS\\Data Analytics Project\\Updated Model")

#Initialize the variables and elements for the model
matplotlib.use("Agg")
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default = 'Sports-Type-Classifier/data',
	help="path to input dataset")
ap.add_argument("-m", "--model", default = 'model/activity.model',
	help="path to output serialized model")
ap.add_argument("-l", "--label-bin", default = 'model/lb.pickle',
	help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

#Initialize the labels that we want to train on
LABELS = set(["weight_lifting", "tennis", "football"])

#Grab the images on which we will train the model
print("Images are now being trained")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]
	if label not in LABELS:
		continue

	#Load and normalize the image to RGB and resize it to 224x224 px
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
    
	data.append(image)
	labels.append(label)

data = np.array(data)
labels = np.array(labels)

#One-Hot Encoding, a part of pre-processing
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#Split as test and train set (standard 25-75 ratio)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

#Initialize and setup the trainer
trainAug = ImageDataGenerator(rotation_range=30, zoom_range=0.15, width_shift_range=0.2,
	height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

valAug = ImageDataGenerator()

#ImageNet setup in RGB
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

#ResNet network to improve efficiency
baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

#The initial model that the main model will be trained on, i.e. add layers to the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

#Final model that will be trained
model = Model(inputs=baseModel.input, outputs=headModel)

#Make sure the layers donot update suring final training
for layer in baseModel.layers:
	layer.trainable = False

#Compile the model
print("Compiling model")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#Train the model for a set number of epochs
print("Training the Model now")
H = model.fit(x=trainAug.flow(trainX, trainY, batch_size=32), steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY), validation_steps=len(testX) // 32, epochs=args["epochs"])

#Evaluate the model efficiency
print("Checking the Model Efficiency")
predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

#Plotting the training efficiency and accuracy(for evaluation purposes)
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

#Save the trained model to hard disk
print("Saving the model for future direct use")
model.save(args["model"], save_format="h5")

#Save the label to hard disk
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
