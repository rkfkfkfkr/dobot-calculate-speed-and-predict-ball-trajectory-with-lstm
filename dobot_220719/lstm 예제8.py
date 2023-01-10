from numpy import array
from keras.models import Sequential
from keras.layers import Dense,LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'ball_data/ball_data1.xlsx'

data = pd.read_excel(filename, engine='openpyxl')

data = np.array(data)

x = []
y = []

plt_x = []
plt_y = []

input_x = []

a1 = int(len(data)/5)

for i in range(a1):

    ax = []

    for j in range(5):

        bx = float(data[5*i+j][1])
        by = float(data[5*i+j][2])

        ax.append([bx,by])
        plt_x.append(bx)
        plt_y.append(by)

    x.append(ax)
    input_x.append(ax)

    if i > 0:

        y.append([float(data[5*i+j][1]),float(data[5*i+j][2])])

y.append([float(data[-1][1]), float(data[-1][2])])
        

x = np.array(x)
y = np.array(y)

print(x)
print('------x reshape--------')
#x = x.reshape(x.shape[0], x.shape[1], 1)
print('x shape : ', x.shape)
print(x)
print(y)

#make_model
model = Sequential()
model.add(LSTM(20, activation = 'relu', input_shape=(5,2)))
model.add(Dense(20))
model.add(Dense(2))

model.summary()

model.compile(optimizer='adam', loss='mse')

#Train
model.fit(x,y,epochs = 100, batch_size =1)

predict_x = []
predict_y = []

for i in range(len(input_x)):

    x_input = np.array(input_x[i])

    x_input = x_input.reshape((1,5,2))

    yhat = model.predict(x_input)

    predict_x.append(float(yhat[0][0]))
    predict_y.append(float(yhat[0][1]))

    #print(predict_x)
    #print(predict_y)

plt.plot(plt_y,plt_x,'b.')
plt.plot(predict_y,predict_x,'r*')
plt.show()

