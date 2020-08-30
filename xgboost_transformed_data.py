# Imports
from xgboost import XGBClassifier
import csv
import joblib
import os.path
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
import matplotlib.pyplot as plt
from numpy import sort
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import random
import csv
import numpy as np
import pandas as pd
import os.path

class MyXGBClassifier(XGBClassifier):
	@property
	def coef_(self):
		return None

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
Funkcija trazi i vraca najcesce vektorske vrijednosti iz danog dataseta
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
Funkcija vraca n podataka koji zadovaoljavaju MAC adresu senzora i priblizni su trazenoj temperaturi
"""
def SearchByMacAndTemperature(mac, temperature, dataset, top):
	tData = []
	#dataset2 = dataset.sort(key = lambda c: c.date) TODO
	for d in dataset:
		if d.mac == mac and abs(temperature - d.temp) < 5 and len(tData) < top:
			tData.append(d)

	return tData

def PrepareXGBoostModel(dataForModelTrain):
	trainX = []
	trainY = []

	for d in dataForModelTrain:
		statusParkinga = float(d.isOcc)
		x1 = float(d.x1)
		y1 = float(d.y1)
		z1 = float(d.z1)
		trainX.append([x1, y1, z1])
		trainY.append(statusParkinga)

	trainX = numpy.array(trainX)
	trainY = numpy.array(trainY)

	model = MyXGBClassifier(max_depth=8, n_estimators=150, learning_rate=0.3)
	model.fit(trainX, trainY)
	print(model)
	return model



def LoadDataFromCsvFile():
	data = []
	reader = csv.reader(open('data/vector_data_nbps.csv', 'r'), delimiter=';')
	# skip columns name
	next(reader)
	for row in reader:
		statusParkinga = int(row[15])
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
		sd = SensorData(sensor_id, mac, x1, y1, z1, x2, y2, z2, temp, statusParkinga, date)
		data.append(sd)
	return data;

def MakePrediction(model, dataForPrediction):
	i=0
	TotalOK=0
	TotalError=0
	for dp in dataForPrediction:

		testX = numpy.array([[dp.x1,dp.y1,dp.z1]])
		predikcija = model.predict(testX)
		if(predikcija != dp.isOcc):
			TotalError+=1
		else:
			TotalOK+=1

	model.get_booster().feature_names = ["X1", "Y1", "Z1", "X2", "Y2", "Z2", "Temperature", "Vector diff", "Sum XYZ"]
	plot_importance(model.get_booster())
	plt.show()

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
	#prepare model
	model = PrepareXGBoostModel(dataForModelTrain)
	#make XGBoost prediction
	MakePrediction(model, dataForPrediction)


if __name__=="__main__": 
    main()
