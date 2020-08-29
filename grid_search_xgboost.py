# Imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import csv
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

data = list()
result= list()

def CalculateBestParametersForXgb():
	model = XGBClassifier()
	n_estimators = [50, 100, 150, 200]
	max_depth = [2, 4, 6, 8]
	param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
	grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
	grid_result = grid_search.fit(np.array(data), np.array(result))
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	# plot results
	scores = numpy.array(means).reshape(len(max_depth), len(n_estimators))
	for i, value in enumerate(max_depth):
		pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
		pyplot.legend()
		pyplot.xlabel('n_estimators')
		pyplot.ylabel('Log Loss')
		pyplot.savefig('n_estimators_vs_max_depth.png')



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
	i=i+1
	if i == 30:
		break;

CalculateBestParametersForXgb()


