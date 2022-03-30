from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow import keras as ke
import numpy as np
import pandas as pd
from matplotlib import pyplot

def labeling(list_):

    one_layers = []
    two_layers = []
    set_one = False
    test_marking = True
    kim = 0
    for i in range(len(list_)):
        st = list_[i]


        if set_one:
            for xi in range(len(one_layers)):
                if st == one_layers[xi]:
                    test_marking = False
                    two_layers.append(xi)
                    break
                else:
                    test_marking = True

        if test_marking:
            one_layers.append(st)
            two_layers.append(kim)
            kim += 1

        set_one = True


    return one_layers, two_layers  #요소, 정수 리스트

data = pd.read_csv("students_performance.csv")
x_data = []
y_data = data[['math score','reading score','writing score']].values.tolist()

a = data[['gender']].values.tolist()
b = data[['parental level of education']].values.tolist()
c = data[['lunch']].values.tolist()
d = data[['test preparation course']].values.tolist()

df = data[['math score']].values.tolist()

aa,aaa = labeling(a)
bb, bbb = labeling(b)
cc, ccc = labeling(c)
dd, ddd = labeling(d)

for i in range(len(aaa)):
    x_data.append([aaa[i],bbb[i],ccc[i],ddd[i]])

model = Sequential([ke.layers.Input(shape=(4)),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(32, activation='relu'),
                    Dropout(0.5),
                    Dense(3, name = 'output')])
print(x_data,y_data)
pyplot.plot(aaa)
pyplot.show()
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
model.fit(x_data, y_data, epochs = 10000)
a = model.predict(x_data[0:1])
print(a)