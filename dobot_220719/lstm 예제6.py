from numpy import array
from keras.models import Sequential
from keras.layers import Dense,LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'ball_data/ball_data1.xlsx'

data = pd.read_excel(filename, engine='openpyxl')

data = np.array(data)

x = np.empty((0,2))
y = np.empty((0,2))

plt_x = []
plt_y = []

for i in range(len(data)):

    ball_x = float(data[i][1])
    ball_y = float(data[i][2])
    ball_v = float(data[i][3])

    plt_x.append(ball_x)
    plt_y.append(ball_y)

    for j in range(len(data)):

        dist = float(data[j][2]) - ball_y

        bx = float(data[j][1])
        by = float(data[j][2])
        bv = float(data[j][3])
        
        if dist > 10:

            y = np.append(y, np.array([[bx,by]]),axis=0)
            x = np.append(x, np.array([[ball_x,ball_y]]),axis=0)

            #plt_x.append(ball_x)
            #plt_y.append(ball_y)

            break
        

print(x)
print('------x reshape--------')
x = x.reshape(x.shape[0], x.shape[1], 1)
print('x shape : ', x.shape)
print(x)

#make_model
model = Sequential()
model.add(LSTM(5, activation = 'relu', input_shape=(2,1)))
model.add(Dense(20))
model.add(Dense(2))

model.summary()

model.compile(optimizer='adam', loss='mse')

#Train
model.fit(x,y,epochs = 50, batch_size =1)

'''
#prediction
x_input = array([36,30])
print(x_input)
x_input = x_input.reshape((1,2,1))

yhat = model.predict(x_input)
print(yhat)
'''

predict_x = []
predict_y = []

for i in range(len(plt_x)):

    i_x = float(plt_x[i])
    i_y = float(plt_y[i])

    x_input = array([i_x,i_y])
    x_input = x_input.reshape((1,2,1))

    yhat = model.predict(x_input)

    p_x = float(yhat[0][0])
    p_y = float(yhat[0][1])

    predict_x.append(p_x)
    predict_y.append(p_y)

plt.plot(plt_y,plt_x,'b.')
plt.plot(predict_y,predict_x,'r*')
plt.show()

