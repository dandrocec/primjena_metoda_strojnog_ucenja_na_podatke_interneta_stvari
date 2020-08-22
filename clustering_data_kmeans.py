# Imports
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

TRAIN_INPUT = list()

#Plot the clusters obtained using k means
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#read data from csv file
reader = csv.reader(open('data/vector_data_nbps.csv', 'r'), delimiter=';')
next(reader)

#take values just from one sensor, MAC: C6:BE:91:04:A8:60, get first 100 rows
i=0
for row in reader:
    if row[16] == "C6:BE:91:04:A8:60" and int(row[13])==1:
        if i < 100:
            state = int(row[15])
            x1 = int(row[5])
            y1 = int(row[6])
            z1 = int(row[7])
            temp = int(row[12])
            TRAIN_INPUT.append([x1, y1, z1])
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

for i in range(0, 5):
    print(*['Possition:',test[i] ,',',test2[i],',',test3[i]])
    sumC=0
    for j in clusts:
        if i==j:
            sumC=sumC+1

    print(*['Poss:',i ,',',sumC])

ax.set_title('K-Means Clustering')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()