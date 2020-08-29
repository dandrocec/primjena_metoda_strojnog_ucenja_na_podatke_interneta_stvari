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
import matplotlib
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

def PrepareXGBoostModel(dataForModelTrain):
	trainX = []
	trainY = []
	for d in dataForModelTrain:
		statusParkinga = int(d.isOcc)
		x1 = int(d.x1)
		y1 = int(d.y1)
		z1 = int(d.z1)
		x2 = int(d.x2)
		y2 = int(d.y2)
		z2 = int(d.z2)
		temp = int(d.temp)
		razlikaVektora = x1 - x2 + y1 - y2 + z1 - z2
		zbrojXYZ = x1+y1+z1
		trainX.append([x1, y1, z1, x2, y2, z2, temp, razlikaVektora, zbrojXYZ])
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
		razlikaVektora = dp.x1 - dp.x2 + dp.y1 - dp.y2 + dp.z1 - dp.z2
		zbrojXYZ = dp.x1+dp.y1+dp.z1
		testX = numpy.array([[dp.x1,dp.y1,dp.z1,dp.x2,dp.y2,dp.z2,dp.temp, razlikaVektora, zbrojXYZ]])
		predikcija = model.predict(testX)
		if(predikcija != dp.isOcc):
			TotalError+=1
		else:
			TotalOK+=1

	#model.get_booster().feature_names = ["X1", "Y1", "Z1", "X2", "Y2", "Z2", "Temperature", "Vector diff", "Sum XYZ"]
	#plot_importance(model.get_booster())
	#plt.show()

	print("Total OK: %d" % TotalOK)
	print("Total error: %d" % TotalError)
	print("Accuracy: %.2f%%" % round(float(TotalOK)/float(len(dataForPrediction))*float(100),2))
	print("Total values for prediction: %d" % len(dataForPrediction))

def main():
	proggressForDataTransform = 1
	transformedData = []
	#load data from CSV file
	rowDataFromSensors = LoadDataFromCsvFile()
	# prepare data for Keras model train
	dataForModelTrain = rowDataFromSensors[:90000]
	print(len(dataForModelTrain))
	# rest of the data will be for prediction
	dataForPrediction = rowDataFromSensors[90000:]
	print(len(dataForPrediction))
	#prepare model
	model = PrepareXGBoostModel(dataForModelTrain)
	#make XGBoost prediction
	MakePrediction(model, dataForPrediction)

if __name__=="__main__": 
    main()
