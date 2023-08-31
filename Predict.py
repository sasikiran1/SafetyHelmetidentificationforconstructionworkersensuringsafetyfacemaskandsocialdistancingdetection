from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import os
import numpy as np
import argparse
import sys
import cv2
from math import pow, sqrt


def detect_helmet(frame,faceNet):
	labelsPath = "model/obj.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	# Initialize list of colors to represent possible class values
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

	# Include paths for YOLO weights and model files
	weightsPath = "model/yolov3-obj_2400.weights"
	configPath = "model/yolov3-obj.cfg"


	# YOLO model for object detector 

	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	(H, W) = frame.shape[:2]
	
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize bounding boxes, confidences,and detecting classes
	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
	
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > 0.5:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence,0.3)
	
		# If found any detection
		if len(idxs) > 0:

			for i in idxs.flatten():
				# extract coordinates for drawing bounding boxes
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
	
				# draw BB and label it
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				print(LABELS[classIDs[i]])
				try:
					crop_img = frame[y:y+h+150, x-10:x+w]
					blob1 = cv2.dnn.blobFromImage(crop_img, 1.0, (224, 224),(104.0, 177.0, 123.0))

					faceNet.setInput(blob1)
					detections1 = faceNet.forward()
					print(detections1.shape)
				
					for j in range(0, detections1.shape[2]):
						confidence1 = detections1[0, 0, j, 2]
						if confidence1 > 0.4:
							print("Confidence",confidence1)
							text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
							cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				except:
					pass
def detect_and_predict_mask(frame, faceNet, maskNet):
	# Get dimensions frame and construct a blob
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))


	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)
prototxtPath = "model/deploy.prototxt"
weightsPath = "model/res10_300x300_ssd_iter_140000.caffemodel"
print(prototxtPath)
print(weightsPath)
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load face mask model 
maskNet = load_model("model/model.h5")

labels = [line.strip() for line in open("model/class_labels.txt")]



bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))


print("\nLoading model...\n")
network = cv2.dnn.readNetFromCaffe("model/SSD_MobileNet_prototxt.txt", "model/SSD_MobileNet.caffemodel")
cap = cv2.VideoCapture("3.mp4")
frame_no = 0
while True:
    ret, frame = cap.read()
    #frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    detect_helmet(frame,faceNet)
    
    network.setInput(blob)
    detections = network.forward()
    pos_dict = dict()
    coordinates = dict()
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        if (mask > withoutMask):
            label = "Mask"
            color = (0, 255, 0)
        elif (withoutMask > mask):
            label = "No Mask"
            color = (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    F = 615
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            if class_id == 15.00:
                cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 2)
               
                label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
                print("{}".format(label))
                coordinates[i] = (startX, startY, endX, endY)
                x_mid = round((startX+endX)/2,4)
                y_mid = round((startY+endY)/2,4)
                height = round(endY-startY,4)

                distance = (165 * F)/height
                print("Distance(cm):{dist}\n".format(dist=distance))

                x_mid_cm = (x_mid * distance) / F
                y_mid_cm = (y_mid * distance) / F
                pos_dict[i] = (x_mid_cm,y_mid_cm,distance)

    close_objects = set()
    for i in pos_dict.keys():
        for j in pos_dict.keys():
            if i < j:
                dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

                # Check distance is less than 200 centimetres
                if dist < 200:
                    close_objects.add(i)
                    close_objects.add(j)

    for i in pos_dict.keys():
        if i in close_objects:
            COLOR = (0,0,255)
        else:
            COLOR = (0,255,0)
        (startX, startY, endX, endY) = coordinates[i]

        cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48,4)), (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
