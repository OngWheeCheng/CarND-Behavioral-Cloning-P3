import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D

epochs = 8 
batchSize = 128
imageWidth = 80
imageHeight = 40
cropTop = 70
cropBottom = 25
steerCorrection = 0.22

# fixed random seed for reproducibility
np.random.seed(epochs)

def crop(image):
    return image[cropTop:-cropBottom, :]

def resize(image, newDim=(imageWidth, imageHeight)):
    return cv2.resize(image, newDim, interpolation=cv2.INTER_AREA)

def flip(image, steerAngle):
	if np.random.rand() < 0.5:
		return np.fliplr(image), -1 * steerAngle
	else:
		return image, steerAngle

def brightness(image):
	min, max = 0.3, 1.0
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to HSV
	v = np.random.uniform(low=min, high=max)
	hsv[:,:,2] = hsv[:,:,2] * v
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def translate(image, angle, rangeX=100, rangeY=10):
    # Randomly shift the image vertically and horizontally (translation)
	transX = rangeX * (np.random.rand() - 0.5)
	transY = rangeY * (np.random.rand() - 0.5)
	angle += transX * 0.002
	transM = np.float32([[1, 0, transX], [0, 1, transY]])
	height, width = image.shape[:2]
	image = cv2.warpAffine(image, transM, (width, height))
	return image, angle
 
def selectImage(center, left, right, steer):
	index = np.random.randint(3)
	if index == 1: # left
		image = cv2.imread(left)
		steer += steerCorrection
	elif index == 2: # right
		image = cv2.imread(right)
		steer -= steerCorrection
	else:
		image = cv2.imread(center)
	return image, steer

def loadData():
	lines = []
	with open('./data/driving_log.csv') as csvfile:
		next(csvfile)  # skip the header
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	ctrImages,leftImages, rightImages, steerAngles = [], [], [], []
	for line in lines:
	    for i in range(3):
	    	sourcePath = line[i]
	    	filename = sourcePath.split('/')[-1]
	    	currentPath = './data/IMG/' + filename
	    	image = cv2.imread(currentPath)
	    	if i == 0:  # images from center camera
	    		ctrImages.append(currentPath)
	    	elif i == 1:  #images from left camera
	    		leftImages.append(currentPath)
	    	else:  #images from right camera
	    		rightImages.append(currentPath)
	    angle = float(line[3])
	    steerAngles.append(angle)
	return np.array(ctrImages), np.array(leftImages), np.array(rightImages), np.array(steerAngles)
  
def buildModel():
	stride2 = (2,2)
	stride1 = (1,1)
	model = Sequential()
	model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(imageHeight, imageWidth, 3)))

	# Convolutional Layers 1 - 3, 5x5 filter, Relu activation function
	model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=stride1, activation='relu', dim_ordering="tf"))
	model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=stride1, activation='relu', dim_ordering="tf"))
	model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=stride2, activation='relu', dim_ordering="tf"))
	
	# Conv Layers 4 - 5, 3x3 filter, Relu activation function
	model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=stride1, activation='relu', dim_ordering="tf"))
	model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=stride1, activation='relu', dim_ordering="tf"))
	
	model.add(Flatten())

	# Fully Connected Layers
	model.add(Dense(100))
	model.add(Dropout(0.2))
	model.add(Dense(50))
	model.add(Dropout(0.2))
	model.add(Dense(10))
	model.add(Dense(1)) # Final FC Layer - just one output - steering angle

	# Show summary of model
	model.summary()
	return model

def augmentImage(image, steer):
	image, angle = flip(image, steer)
	image, angle = translate(image, angle)
	image = brightness(image)
	image = resize(crop(image))
	return image, angle

def dataGenerator(center, left, right, steer, isTrain, bsize=64):
    while 1:
    	batchImages = []
    	batchAngles = []
    	ctrImg, leftImg, rightImg, steerAngle = shuffle(center, left, right, steer)
    	for i in np.random.randint(low=0, high=len(ctrImg), size=bsize):
    		keep, threshold = 0, 0.95
    		while keep == 0:
    			image, angle = selectImage(ctrImg[i], leftImg[i], rightImg[i], steerAngle[i])
    			image, angle = augmentImage(image, angle)
    			if abs(angle) < 0.1:
    				val = np.random.uniform()
    				if val > threshold:
    					keep = 1
    			else:
    				keep = 1
                    
    		batchImages.append(image)
    		batchAngles.append(angle)
    	yield np.array(batchImages), np.array(batchAngles)

def main():
	# read input data - images and steering angles	
	ctrData, leftData, rightData, steerData = loadData()
	
	# shuffle first, then split into training and validation sets
	ctrImage, leftImage, rightImage, steerAngle = shuffle(ctrData, leftData, rightData, steerData)
	ctrTrain, ctrValid, leftTrain, leftValid, rightTrain, rightValid, steerTrain, steerValid = train_test_split(ctrImage, leftImage, rightImage, steerAngle, test_size=0.1)
	print("ctrTrain = ", len(ctrTrain), " ctrValid = ", len(ctrValid))
	print("Train angles = ", len(steerTrain), " Validation angles = ", len(steerValid))

	model = buildModel()

	# Compile and train the model
	model.compile(loss='mse', optimizer=Adam(lr = 0.0001), metrics=['accuracy'])

	trainGenerator = dataGenerator(ctrTrain, leftTrain, rightTrain, steerTrain, 1)
	validGenerator = dataGenerator(ctrValid, leftValid, rightValid, steerValid, 0)
	model.fit_generator(trainGenerator,
			samples_per_epoch = len(ctrTrain),
			nb_epoch = epochs,
			validation_data=validGenerator,
			nb_val_samples=len(ctrValid), 
			verbose=1)
	
	# Save model architecture and weights
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights('model.h5')

if __name__ == '__main__':
	main()