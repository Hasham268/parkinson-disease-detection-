from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from imutils import paths
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix
from imutils import build_montages
from sklearn import metrics as m

def quantify_image(image):
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")
	return features

def load_split(path):
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []
	
	for imagePath in imagePaths:
		label = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))		
		image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		features = quantify_image(image)
		data.append(features)
		labels.append(label)
	return (np.array(data), np.array(labels))

trainingPath = os.path.sep.join(["dataset/spiral", "training"])
testingPath = os.path.sep.join(["dataset/spiral", "testing"])

(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)
trials = {}

for i in range(0, 5):

	print("training model {} of {}...".format(i + 1,5))
	model = RandomForestClassifier(n_estimators=100)
	model.fit(trainX, trainY)

	predictions = model.predict(testX)

	metrics = {}
	cm = confusion_matrix(testY, predictions).flatten()
	(tn, fp, fn, tp) = cm
	metrics["ACCURACY"] = ((tp + tn) / float(cm.sum())) * 100
	metrics["RECALL / SENSITIVITY / TRUE POSITIVE RATE"] = (tp / float(tp + fn)) * 100
	metrics["SPECIFICITY / TRUE NEGATIVE RATE"] = (tn / float(tn + fp)) * 100
	metrics["PRECISION"] = (tp / float(tp + fp)) * 100

	for (k, v) in metrics.items():
		l = trials.get(k, [])
		l.append(v)
		trials[k] = l

print()
print(predictions)
print(testY)
    
print()
for (metric, value) in metrics.items():
	print(metric, " = ", value)

print()
for metric in ("ACCURACY", "RECALL / SENSITIVITY / TRUE POSITIVE RATE", "SPECIFICITY / TRUE NEGATIVE RATE", "PRECISION"):
	values = trials[metric]
	mean = np.mean(values)
	std = np.std(values)
	print(metric)
	print("=" * len(metric))
	print("u={:.4f}, o={:.4f}".format(mean, std))
	print("")

print()
print("MAE = ", m.mean_absolute_error(testY, predictions))
print("MSE = ", m.mean_squared_error(testY, predictions))
print("RMSE = ", np.sqrt(m.mean_squared_error(testY, predictions)))

testingPaths = list(paths.list_images(testingPath))
idxs = np.arange(0, len(testingPaths))
idxs = np.random.choice(idxs, size=(25,), replace=False)

images = []

for i in idxs:
	image = cv2.imread(testingPaths[i])
	output = image.copy()
	output = cv2.resize(output, (128, 128))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (200, 200))
	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	features = quantify_image(image)
	preds = model.predict([features])    
	label = le.inverse_transform(preds)[0]

	color = (255, 0, 0) if label == "healthy" else (0, 0, 255)
	cv2.putText(output, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	images.append(output)

montage = build_montages(images, (150, 150), (8, 3))[0]
cv2.imshow("GUI REPRESENTATION OF PROJECT", montage)
cv2.waitKey(0)









