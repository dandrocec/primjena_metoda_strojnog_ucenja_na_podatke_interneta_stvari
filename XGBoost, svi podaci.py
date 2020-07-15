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
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy

# define custom class to fix bug in xgboost 1.0.2
class MyXGBClassifier(XGBClassifier):
	@property
	def coef_(self):
		return None

ULAZNI_PODACI_ZA_MODEL = list()
ULAZNI_PODACI_ZA_MODEL_DETEKCIJA= list()
PODACI_ZA_TESTIRANJE= list()
PODACI_ZA_TESTIRANJE_DETEKCIJA= list()

modelFileName = 'xgboost_model'

def main():
    model = pripremi_model()
    testiraj_predikciju(model)

def procitaj_podatke_iz_datoteke():
    reader = csv.reader(open('data/vector_data_nbps.csv', 'r'), delimiter=';')
    # preskoči nazive kolona
    next(reader)
    i=0
    for row in reader:
        statusParkinga = int(row[15])
        x1 = int(row[5])
        y1 = int(row[6])
        z1 = int(row[7])
        x2 = int(row[8])
        y2 = int(row[9])
        z2 = int(row[10])
        temperatura = int(row[12])
        razlikaVektora = x1 - x2 + y1 - y2 + z1 - z2
        zbrojXYZ = x1+y1+z1

        if i < 90000 and int(row[13])==1:
            # kreiraj model s 9000 podataka
            ULAZNI_PODACI_ZA_MODEL.append([x1, y1, z1, x2, y2, z2, temperatura, razlikaVektora, zbrojXYZ])
            ULAZNI_PODACI_ZA_MODEL_DETEKCIJA.append(statusParkinga)
            i=i+1
        else:
            # pripremi podatke za testiranje, ovi podaci se ne nalaze u kreiranom modelu
            PODACI_ZA_TESTIRANJE.append([x1, y1, z1, x2, y2, z2, temperatura, razlikaVektora, zbrojXYZ])
            PODACI_ZA_TESTIRANJE_DETEKCIJA.append(statusParkinga)

def testiraj_predikciju(model):
    UkupnoTocno=0
    UkupnoNetocno=0
    #podesite_broj_stabala_i_najvecu_dubinu()

    for x in range(0, len(PODACI_ZA_TESTIRANJE_DETEKCIJA)):
        TEST = list()
        TEST.append(PODACI_ZA_TESTIRANJE[x])
        predikcija = model.predict(np.array(TEST))
        if(predikcija != PODACI_ZA_TESTIRANJE_DETEKCIJA[x]):
            UkupnoNetocno+=1
        else:
            UkupnoTocno+=1
    

    model.get_booster().feature_names = ["X1", "Y1", "Z1", "X2", "Y2", "Z2", "Temperatura", "Razlika vektora", "Zbroj XYZ"]
    plot_importance(model.get_booster())
    plt.show()

    print("Ukupno tocnih: %d" % UkupnoTocno)
    print("Ukupno netocnih: %d" % UkupnoNetocno)
    print("Postotak tocnosti: %.2f%%" % round(float(UkupnoTocno)/float(len(PODACI_ZA_TESTIRANJE_DETEKCIJA))*float(100),2))
    print("Ukupno testiranih vrijednosti: %d" % len(PODACI_ZA_TESTIRANJE_DETEKCIJA))

def provjeri_vaznost_znacajki(model):
    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:
        # odaberi znacajke koristenjem granica
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(ULAZNI_PODACI_ZA_MODEL)
        # treniraj model
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, ULAZNI_PODACI_ZA_MODEL_DETEKCIJA)
        select_X_test = selection.transform(PODACI_ZA_TESTIRANJE)
        #odradi predikciju
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(PODACI_ZA_TESTIRANJE_DETEKCIJA, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

def podesi_broj_stabala_i_najvecu_dubinu():
    model = XGBClassifier()
    n_estimators = [50, 100, 150, 200]
    max_depth = [2, 4, 6, 8]
    print(max_depth)
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(np.array(ULAZNI_PODACI_ZA_MODEL), np.array(ULAZNI_PODACI_ZA_MODEL_DETEKCIJA))
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

def pripremi_model():
    model = MyXGBClassifier(max_depth=8, n_estimators=150, learning_rate=0.3)
    procitaj_podatke_iz_datoteke()
    model.fit(np.array(ULAZNI_PODACI_ZA_MODEL), np.array(ULAZNI_PODACI_ZA_MODEL_DETEKCIJA))
    print(model)
    return model

# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 