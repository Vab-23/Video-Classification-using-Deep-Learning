#Import relevant Libraries
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import os

#Change the directory location to where the model is saved
os.chdir("E:\\Sem 5\\PROJECTS\\Data Analytics Project\\Updated Model")


#Make the argument arser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default = 'model/activity.model',
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", default = 'model/lb.pickle',
	help="path to  label binarizer")
ap.add_argument("-i", "--input", default = 'example_clips/main.mp4',
	help="path to our input video")
ap.add_argument("-o", "--output", default = 'output/lifting.avi',
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

#Load the trained model from the hard disk, that was saved by the trainer
print("Loading the pre-saved trained model")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

#Initialize the image and mean subtarct it
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

#Initialize the video stream, output video file
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

#Loop over the frames from the video
while True:
    #Grab the video frame
	(grabbed, frame) = vs.read()

	#If no frame exists, the video has ended and so break the loop
	if not grabbed:
		break

	#If the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	#Normalization of the loaded frame, convert to RGB and resize to 224x224 px and perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean

	#Make the predicitons on the frame
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]

	#Draw the activity name on the frame
	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)

	#Check if video writer is not initialized
	if writer is None:
		#Initialize the video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	#Save the edited frames in a video to the hard disk
	writer.write(output)

	#Show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	#If the break key is pressed, quit the loop
	if key == ord("q"):
		break

#Release all of the opened frames and windows
print("Shutting down dependencies..")
writer.release()
vs.release()
