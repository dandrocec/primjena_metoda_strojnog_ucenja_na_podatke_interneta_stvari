# Imports
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# Load Data
iris = load_iris()

# Create a dataframe
#df = pd.DataFrame(iris.data, columns = iris.feature_names)
#df['target'] = iris.target
#X = iris.data
#df.sample(4)

TRAIN_INPUT = list()
TRAIN_OUTPUT= list()

#Plot the clusters obtained using k means
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

reader = csv.reader(open('data/vector_data_nbps.csv', 'r'), delimiter=';')
next(reader)
i=0
for row in reader:
    if row[16] == "C6:BE:91:04:A8:60" and int(row[13])==1:
        if i < 100:
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
            diff2 = x1 - int(-482) + y1 - int(443) + z1 - int(-9)
            TRAIN_INPUT.append([x1, y1, z1])
            #TTRAIN_INPUT.append([x1, y1, z1])
            TRAIN_OUTPUT.append(state)

            if state == 0:
                ax.scatter(x1, y1, z1,c='g')
            else:
                ax.scatter(x1, y1, z1,c='r')

        i=i+1



# Instantiate Kmeans
km = KMeans(6)
clusts = km.fit_predict(np.array(TRAIN_INPUT))

test = km.cluster_centers_[:, 0]
test2 = km.cluster_centers_[:, 1]
test3 = km.cluster_centers_[:, 2]

scatter = ax.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            km.cluster_centers_[:, 2],
            s = 250,
            marker='o',
            c='red',
            label='centroids')
#scatter = ax.scatter(df['petal width (cm)'],df['sepal length (cm)'], df['petal length (cm)'],
#                     c=clusts,s=20, cmap='winter')



for i in range(0, 5):
    print(*['Possition:',test[i] ,',',test2[i],',',test3[i]])
    sumC=0
    for j in clusts:
        if i==j:
            sumC=sumC+1

    print(*['Poss:',i ,',',sumC])
    


    #print("Possition: %d, %d, %d" % test[i], test2[i], test3[i])

ax.set_title('K-Means Clustering')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()