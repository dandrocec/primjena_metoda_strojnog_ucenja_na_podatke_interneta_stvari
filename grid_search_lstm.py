# Imports
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import numpy
import csv
import os.path
from numpy import array


def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape=(1, 3)))
	model.add(LSTM(50))
	model.add(Dense(10))
	model.add(Dropout(0.3))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
	return model


data = list()
result= list()

reader = csv.reader(open('data/temp_nbps.csv', 'r'), delimiter=',')
next(reader)
i=0
for row in reader:
	statusParkinga = float(row[6])
	x1 = float(row[3])
	y1 = float(row[4])
	z1 = float(row[5])

	data.append([x1, y1, z1])
	result.append(statusParkinga)


seed = 7
numpy.random.seed(seed)
model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
activation1 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)

trainX = numpy.array(data)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
grid_result = grid.fit(trainX, numpy.array(result), verbose=2)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f & %f & %r" % (mean, stdev, param))