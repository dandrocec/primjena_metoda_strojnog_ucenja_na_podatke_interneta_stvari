from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import numpy
import csv
import joblib
import os.path
from tensorflow.keras.models import model_from_json
from numpy import array
import numpy as np
import matplotlib.pyplot as plt

ULAZNI_PODACI_ZA_MODEL = list()
ULAZNI_PODACI_ZA_MODEL_DETEKCIJA= list()
PODACI_ZA_TESTIRANJE= list()
PODACI_ZA_TESTIRANJE_DETEKCIJA= list()
modelFileName = 'KerasModelNew.json'

def main():
    procitaj_podatke_iz_datoteke()
    model = pripremi_model()
    testiraj_predikciju(model)

def procitaj_podatke_iz_datoteke():
    reader = csv.reader(open('data/jedan senzor 18000 podataka.csv', 'r'), delimiter=';')
    # preskoči nazive kolona
    next(reader)
    i=0
    for row in reader:
        statusParkinga = float(row[15])
        x1 = float(row[5])
        y1 = float(row[6])
        z1 = float(row[7])
        x2 = float(row[8])
        y2 = float(row[9])
        z2 = float(row[10])
        temperatura = float(row[12])
        razlikaVektora = x1 - x2 + y1 - y2 + z1 - z2
        zbrojXYZ = x1+y1+z1

        if i < 15000 and float(row[13])==1:
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

    #scaler = MinMaxScaler(feature_range=(0, 1))
    #dataset = numpy.array(PODACI_ZA_TESTIRANJE_DETEKCIJA)
    #dataset = scaler.fit_transform(dataset)
    #trainY = numpy.reshape(dataset, (dataset.shape[0], 1, dataset.shape[1]))
    DV = numpy.array(PODACI_ZA_TESTIRANJE)

    for x in range(0, len(DV)):
        #TEST = list()
        #TEST.append(trainY[x])
        TT = [[PODACI_ZA_TESTIRANJE[x]]]
        print(TT)
        predikcija = model.predict(numpy.array(TT))
        print(PODACI_ZA_TESTIRANJE_DETEKCIJA[x])
        print(predikcija)
        if(predikcija != PODACI_ZA_TESTIRANJE_DETEKCIJA[x]):
            UkupnoNetocno+=1
        else:
            UkupnoTocno+=1


    #model.get_booster().feature_names = ["X1", "Y1", "Z1", "X2", "Y2", "Z2", "Temperatura", "Razlika vektora", "Zbroj XYZ"]
    #plot_importance(model.get_booster())
    #plt.show()

    print("Ukupno tocnih: %d" % UkupnoTocno)
    print("Ukupno netocnih: %d" % UkupnoNetocno)
    print("Postotak tocnosti: %.2f%%" % round(float(UkupnoTocno)/float(len(PODACI_ZA_TESTIRANJE_DETEKCIJA))*float(100),2))
    print("Ukupno testiranih vrijednosti: %d" % len(PODACI_ZA_TESTIRANJE_DETEKCIJA))

def pripremi_model():
    if os.path.isfile(modelFileName):
        json_file = open(modelFileName, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("KerasModelNew.h5")
        print("Model je ucitan iz diska")
        return loaded_model
    else:
        #model = Sequential()
        #model.add(Dense(16, input_dim=9, activation='relu', kernel_initializer='uniform'))
        #model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
        #model.add(Dense(8, activation='softplus', kernel_initializer='uniform'))
        #model.add(Dense(1, activation='softplus', kernel_initializer='uniform'))
        #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        #train_X = ULAZNI_PODACI_ZA_MODEL.reshape((ULAZNI_PODACI_ZA_MODEL.shape[0], 24, 9))
        #train_y = 1
        
        print(ULAZNI_PODACI_ZA_MODEL)
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        dataset = numpy.array(ULAZNI_PODACI_ZA_MODEL)
        print(dataset)
        dataset = scaler.fit_transform(dataset)
        print(dataset)
        trainX = numpy.reshape(dataset, (dataset.shape[0], 1, dataset.shape[1]))
        print(trainX)
        #testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        # create and fit the LSTM network
        look_back = 1
        model = Sequential()
        model.add(Dense(64, input_shape=(1, 9), activation='relu', kernel_initializer='uniform'))
        model.add(LSTM(64, activation='softplus', kernel_initializer='uniform'))
        #model.add(Dropout(0.2))
        model.add(Dense(1, activation='softplus', kernel_initializer='uniform'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        #model.compile(loss='mae', optimizer='adam')
        #print(model.summary())
        # fit network
        print(ULAZNI_PODACI_ZA_MODEL_DETEKCIJA)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        model.fit(trainX, numpy.array(ULAZNI_PODACI_ZA_MODEL_DETEKCIJA), epochs=20, batch_size=5, verbose=2)
        print(model.summary())
        # plot history
        #model.fit(array(ULAZNI_PODACI_ZA_MODEL), array(ULAZNI_PODACI_ZA_MODEL_DETEKCIJA), epochs=20, batch_size=5)
        model_json = model.to_json()
        with open(modelFileName, "w") as json_file:
            json_file.write(model_json)
            model.save_weights("KerasModelNew.h5")
            print("Model je spremljen na disk")
        
        return model


if __name__=="__main__": 
    main() 