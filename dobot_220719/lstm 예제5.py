from numpy import array
from keras.models import Sequential
from keras.layers import Dense,LSTM

#data
x = array([[1,1,2], [3,3,2], [5,5,2], [7,7,1], [8,8,1], [9,9,1]])
y = array([[5,5,2], [7,7,2], [8,8,1], [9,9,1], [10,10,1], [11,11,1]])

print('x shape: ', x.shape)
print('y shape: ', y.shape)

print(x)
print('------x reshape--------')
x = x.reshape(x.shape[0], x.shape[1], 1)
print('x shape : ', x.shape)
print(x)

#make_model
model = Sequential()
model.add(LSTM(5, activation = 'relu', input_shape=(3,1)))
model.add(Dense(5))
model.add(Dense(3))

model.summary()

model.compile(optimizer='adam', loss='mse')

#Train
model.fit(x,y,epochs = 50, batch_size =1)


#prediction

x_input = array([10,10,1])
print(x_input)
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)
print(yhat[0][0])
