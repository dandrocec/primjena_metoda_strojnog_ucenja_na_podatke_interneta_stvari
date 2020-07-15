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

modelFileName = 'KerasModelNew.json'

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


def PrepareKerasModel(dataForModelTrain):

	if os.path.isfile(modelFileName):
		json_file = open(modelFileName, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("KerasModelNew.h5")
		print("Model prepared, loaded from disk")
		return loaded_model
	else:
		trainX = []
		trainY = []

		for d in dataForModelTrain:
			statusParkinga = float(d.isOcc)
			razlikaVektora = d.x1 - d.x2 + d.y1 - d.y2 + d.z1 - d.z2
			trainX.append([d.x1, d.y1, d.z1, d.x2, d.y2, d.z2, d.temp, razlikaVektora])
			trainY.append(statusParkinga)

		trainX = numpy.array(trainX)
		trainY = numpy.array(trainY)
		trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

		model = Sequential()
		model.add(LSTM(50, return_sequences=True, input_shape=(1, 8)))
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
		# for test
		if(len(data)>2000):
			break
	return data;

def MakePrediction(model, dataForPrediction):
	i=0
	tocno=0;
	netocno=0
	for dp in dataForPrediction:
		statusParkinga = int(dp.isOcc)
		razlikaVektora = dp.x1 - dp.x2 + dp.y1 - dp.y2 + dp.z1 - dp.z2
		testX = numpy.array([[[dp.x1, dp.y1, dp.z1, dp.x2, dp.y2, dp.z2, dp.temp, razlikaVektora]]])
		trainPredict = model.predict(testX)

		p = float(trainPredict[0][0])
		print("Parking event: %d %d %d %d %f" % (dp.x1,dp.y1,dp.z1, statusParkinga, p))
		
		if(float(p) > float(0.50) and statusParkinga == 1):
			tocno+=1
		elif (float(p) > float(0.50) and statusParkinga == 0):
			netocno+=1
			print("Prediction error: %d %d %d %d %f" % (dp.x1,dp.y1,dp.z1, statusParkinga, p))
		elif (float(p) <= float(0.50) and statusParkinga == 0):
			tocno+=1
		elif (float(p) > float(0.50) and statusParkinga == 0):
			netocno+=1
			print("Prediction error: %d %d %d %d %f" % (dp.x1,dp.y1,dp.z1, statusParkinga, p))
		elif (float(p) < float(0.50) and statusParkinga == 1):
			netocno+=1
			print("Prediction error: %d %d %d %d %f" % (dp.x1,dp.y1,dp.z1, statusParkinga, p))


	print(tocno)
	print(netocno)
	print(i)

def main():
	proggressForDataTransform = 1
	transformedData = []
	rowDataFromSensors = LoadDataFromCsvFile()

	# prepare data for Keras model train
	dataForModelTrain = rowDataFromSensors[:1500]
	# rest of the data will be for prediction
	dataForPrediction = rowDataFromSensors[1500:]

	model = PrepareKerasModel(dataForModelTrain)

	MakePrediction(model, dataForPrediction)

	#np_array = np.array(transformedData)
	#pd.DataFrame(np_array).to_csv("temp_nbps.csv")


if __name__=="__main__": 
    main()