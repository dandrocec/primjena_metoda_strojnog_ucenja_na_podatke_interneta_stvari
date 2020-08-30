# Imports
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import random
import csv
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json
import os.path

modelFileName = 'KerasModelTansData.json'

class SensorData:
	def __init__(self, id, mac, x1,y1,z1,x2,y2,z2, temp, isOcc, date):
		self.id = id
		self.mac = mac
		self.x1 = x1
		self.y1 = y1
		self.z1 = z1
		self.x2 = x2
		self.y2 = y2
		self.z2 = z2
		self.temp = temp;
		self.isOcc = isOcc
		self.date = date

class TransformedData:
	def __init__(self, mac, x1, y1, z1, isOcc, date):
		self.mac = mac
		self.x1 = x1
		self.y1 = y1
		self.z1 = z1
		self.isOcc = isOcc
		self.date = date

class VectorData:
	def __init__(self,x1,y1,z1):
		self.x1 = x1
		self.y1 = y1
		self.z1 = z1

"""
The function searches and returns the most common vector values from a given dataset
"""
def SearchForMostCommonValues(dataset, limitForSimilarValue):

	mostCommonValueX = 0
	maxSimilarX = 0
	mostCommonValueY = 0
	maxSimilarY = 0
	mostCommonValueZ = 0
	maxSimilarZ = 0

	for d1 in dataset:
		similarX = 0
		similarY = 0
		similarZ = 0
		for d2 in dataset:
			if abs(d1.x1 - d2.x1) < limitForSimilarValue:
				similarX+=1
			if abs(d1.y1 - d2.y1) < limitForSimilarValue:
				similarY+=1
			if abs(d1.z1 - d2.z1) < limitForSimilarValue:
				similarZ+=1
		if similarX > maxSimilarX:
			maxSimilarX = similarX
			mostCommonValueX = d1.x1
		if similarY > maxSimilarY:
			maxSimilarY = similarY
			mostCommonValueY = d1.y1
		if similarZ > maxSimilarZ:
			maxSimilarZ = similarZ
			mostCommonValueZ = d1.z1
	return VectorData(mostCommonValueX, mostCommonValueY, mostCommonValueZ);

"""
Search data by MAC and by temperature
"""
def SearchByMacAndTemperature(mac, temperature, dataset, top):
	tData = []
	for d in dataset:
		if d.mac == mac and abs(temperature - d.temp) < 5 and len(tData) < top:
			tData.append(d)

	return tData

"""
Create or load Keras model
"""
def PrepareKerasModel(dataForModelTrain):

	if os.path.isfile(modelFileName):
		json_file = open(modelFileName, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("KerasModelTansData.h5")
		print("Model prepared, loaded from disk")
		return loaded_model
	else:
		trainX = []
		trainY = []

		for d in dataForModelTrain:
			parkingStatus = float(d.isOcc)
			x1 = float(d.x1)
			y1 = float(d.y1)
			z1 = float(d.z1)
			trainX.append([x1, y1, z1])
			trainY.append(parkingStatus)

		trainX = numpy.array(trainX)
		trainY = numpy.array(trainY)
		trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

		model = Sequential()
		model.add(LSTM(50, return_sequences=True, input_shape=(1, 3)))
		model.add(LSTM(50))
		model.add(Dense(10))
		model.add(Dropout(0.3))
		model.add(Dense(1))
		model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
		model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
		
		print(model.summary())

		model_json = model.to_json()
		with open(modelFileName, "w") as json_file:
			json_file.write(model_json)
			model.save_weights("KerasModelNew.h5")
			print("Model saved.")

		return model

"""
Load data from .csv file, and return object with values
"""
def LoadDataFromCsvFile():
	data = []
	reader = csv.reader(open('data/vector_data_nbps.csv', 'r'), delimiter=';')
	# skip columns name
	next(reader)
	for row in reader:
		parkingStatus = int(row[15])
		sensor_id = int(row[0])
		mac = row[16]
		x1 = float(row[5])
		y1 = float(row[6])
		z1 = float(row[7])
		x2 = float(row[8])
		y2 = float(row[9])
		z2 = float(row[10])
		temp = float(row[12])
		date = row[1]
		sd = SensorData(sensor_id, mac, x1, y1, z1, x2, y2, z2, temp, parkingStatus, date)
		data.append(sd)
	return data;

"""
Test Keras prediction and calculate accuracy
"""
def MakePrediction(model, dataForPrediction):
	i=0
	TotalOK=0;
	TotalError=0
	for dp in dataForPrediction:
		parkingStatus = int(dp.isOcc)
		testX = numpy.array([[[dp.x1,dp.y1,dp.z1]]])
		trainPredict = model.predict(testX)

		p = float(trainPredict[0][0])
		print("Parking event: %d %d %d %d %f" % (dp.x1,dp.y1,dp.z1, parkingStatus, p))
		
		if(float(p) > float(0.65) and parkingStatus == 1):
			TotalOK+=1
		elif (float(p) > float(0.65) and parkingStatus == 0):
			TotalError+=1
			print("Prediction error: %d %d %d %d %f" % (dp.x1,dp.y1,dp.z1, parkingStatus, p))
		elif (float(p) <= float(0.65) and parkingStatus == 0):
			TotalOK+=1
		elif (float(p) > float(0.65) and parkingStatus == 0):
			TotalError+=1
			print("Prediction error: %d %d %d %d %f" % (dp.x1,dp.y1,dp.z1, parkingStatus, p))
		elif (float(p) < float(0.65) and parkingStatus == 1):
			TotalError+=1
			print("Prediction error: %d %d %d %d %f" % (dp.x1,dp.y1,dp.z1, parkingStatus, p))

	print("Total OK: %d" % TotalOK)
	print("Total error: %d" % TotalError)
	print("Accuracy: %.2f%%" % round(float(TotalOK)/float(len(dataForPrediction))*float(100),2))
	print("Total values for prediction: %d" % len(dataForPrediction))

def main():
	proggressForDataTransform = 1
	transformedData = []
	rowDataFromSensors = LoadDataFromCsvFile()
	# transform row data from sensor
	for row in rowDataFromSensors:	
		tData = SearchByMacAndTemperature(row.mac, row.temp, rowDataFromSensors, 200)

		test = SearchForMostCommonValues(tData, 20)

		transformedData.append(TransformedData(row.mac, row.x1-test.x1, row.y1-test.y1, row.z1-test.z1, row.isOcc, row.date))

		if(proggressForDataTransform % 100 == 0):
			print("Transformation data progress: %d / %d, %d percentage" % (proggressForDataTransform, len(rowDataFromSensors), float(proggressForDataTransform) / float(len(rowDataFromSensors)) * float(100)))
		proggressForDataTransform+=1
	
	# prepare data for Keras model train
	dataForModelTrain = transformedData[:90000]
	# rest of the data will be for prediction
	dataForPrediction = transformedData[90000:]
	# #prepare model
	model = PrepareKerasModel(dataForModelTrain)
	#make Keras prediction
	MakePrediction(model, dataForPrediction)


if __name__=="__main__": 
    main()
