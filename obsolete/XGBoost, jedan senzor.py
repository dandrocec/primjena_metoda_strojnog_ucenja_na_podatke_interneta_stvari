from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import joblib
import os.path
import xgboost as xgb
import numpy as np
from numpy import reshape
from numpy import array
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from xgboost import plot_importance
import matplotlib.pyplot as plt

# Create an empty list of the input training set 'X' and create an empty list of the output for each training set 'Y'
TRAIN_INPUT = list()
TRAIN_OUTPUT= list()

modelFileName = 'xgboost_model_jedan_senzor'


def main():
    model = prepare_model()
    test_data(model)



def read_data_from_file():
    reader = csv.reader(open('data/jedan senzor 18000 podataka.csv', 'r'), delimiter=';')
    next(reader)
    i=0
    lastPredition = int(1)
    for row in reader:
        if i < 15000:
            state = int(row[15])
            x1 = int(row[5])
            y1 = int(row[6])
            z1 = int(row[7])
            x2 = int(row[8])
            y2 = int(row[9])
            z2 = int(row[10])
            temp = int(row[12])

            diff = x1 - x2 + y1 - y2 + z1 - z2
            sumV = x1+y1+z1
            TRAIN_INPUT.append([x1, y1, z1, x2, y2, z2, temp, diff, sumV])
            TRAIN_OUTPUT.append(state)
            lastPredition = state
        i=i+1



def test_data(model):

    # plot feature importance
    plot_importance(model)
    pyplot.show()

    reader = csv.reader(open('data/jedan senzor 18000 podataka.csv', 'r'), delimiter=';')
    next(reader)

    ok=0
    error=0
    errorOcc=0
    errorFree=0
    i=0
    total=0
    lastPredition = int(1)
    X = []
    Y = []
    Z = []
    X2 = []
    Y2 = []
    Z2 = []

    Xok = []
    Yok = []
    Zok = []

    for row in reader:
        if i > 15000:
            TEST = list()
            TEST_RESULT = list()
            state = int(row[15])
            x1 = int(row[5])
            y1 = int(row[6])
            z1 = int(row[7])
            x2 = int(row[8])
            y2 = int(row[9])
            z2 = int(row[10])
            temp = int(row[12])
            diff = x1 - x2 + y1 - y2 + z1 - z2
            sumV = x1+y1+z1
            TEST.append([x1, y1, z1, x2, y2, z2, temp, diff, sumV])

            TEST_RESULT.append(state)
            y_pred = model.predict(np.array(TEST))
            lastPredition = y_pred
            total += 1

            if y_pred==1 and state==0:
                errorFree+=1
                X2.append(x1)
                Y2.append(y1)
                Z2.append(z1)
            
            if y_pred==0 and state==1:
                errorOcc+=1
                X.append(x1)
                Y.append(y1)
                Z.append(z1)


            if y_pred==1 and state==1:
                ok=ok+1
            elif y_pred==0 and state==0:
                ok=ok+1
                Xok.append(x1)
                Yok.append(y1)
                Zok.append(z1)
            else:
                error+=1
        i=i+1

    if error < ok:
        print("Accuracy: %.2f%%" % (100 - (error/total * 100.0)))

    print("Error occ: %d" % errorOcc)
    print("Error free: %d" % errorFree)
    print("Total: %d" % i)

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z,c='r')
    ax.scatter(X2, Y2, Z2,c='g')
    ax.scatter(Xok, Yok, Zok,c='b')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    pyplot.title("Podaci od jednog senzora", fontsize=14)

    pyplot.show()


def prepare_model():
    #if os.path.isfile(modelFileName):
        # load the model from disk
        #model = joblib.load(modelFileName)
        #print(model)
        #return model
    #else:
        # Train the model on training data
        # save the model to disk
    model = XGBClassifier()
    read_data_from_file()
    model.fit(np.array(TRAIN_INPUT), np.array(TRAIN_OUTPUT))
    joblib.dump(model, modelFileName)
    return model


# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 