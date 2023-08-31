from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os



datasetpath="dataset/"

learn_rate = 1e-4
EPOCHS = 20
batch_size = 32

print("[INFO] loading images...")
imagePaths = list(paths.list_images(datasetpath))
print(imagePaths)
data = []
labels = []

for imagePath in imagePaths:
	print(imagePath)
	label = imagePath.split(os.path.sep)[-2]

	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	data.append(image)
	labels.append(label)

# Converting data and labels to NumPy array values
data = np.array(data, dtype="float32")
labels = np.array(labels)

# one-hot encoding on the labels to make it as binary value
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

mModel = baseModel.output
mModel = AveragePooling2D(pool_size=(7, 7))(mModel)
mModel = Flatten(name="flatten")(mModel)
mModel = Dense(128, activation="relu")(mModel)
mModel = Dropout(0.5)(mModel)
mModel = Dense(2, activation="softmax")(mModel)

model = Model(inputs=baseModel.input, outputs=mModel)

for layer in baseModel.layers:
	layer.trainable = False

print("[INFO] compiling model...")
opt = Adam(lr=learn_rate, decay=learn_rate / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

print("[INFO] training head...")
history = model.fit(
	aug.flow(trainX, trainY, batch_size=batch_size),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=batch_size)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

print("[INFO] saving mask detector model...")
model.save("results/model.h5")

N = EPOCHS
plt.clf()   # clear figure
plt.plot(np.arange(0, N), history.history["loss"], label="loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/Loss.png') 
plt.pause(5)
plt.show(block=False)
plt.close()	

plt.clf()   # clear figure
acc_values = history.history['accuracy']
val_acc_values = history.history['val_accuracy']
plt.plot(np.arange(0, N), acc_values, label='Training acc')
plt.plot(np.arange(0, N), val_acc_values, label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/Accuracy.png') 
plt.pause(5)
plt.show(block=False)
plt.close()	
