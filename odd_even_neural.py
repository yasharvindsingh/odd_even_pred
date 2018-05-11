import numpy as np 
import pandas as pd 
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from random import shuffle
from sklearn.model_selection import train_test_split

df = pd.read_csv('odd_even_data.csv')
X = np.array(df['num'].values)
y = np.array(df['label'].values)

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size = 0.1)

model = Sequential()
model.add(Dense(10,input_dim=1,activation='relu', bias_initializer=keras.initializers.Ones()))
model.add(BatchNormalization())
model.add(Dense(20,activation='relu',bias_initializer=keras.initializers.Ones()))
model.add(Dense(10,activation='relu',bias_initializer=keras.initializers.Ones()))
model.add(BatchNormalization())
model.add(Dense(20,activation='relu',bias_initializer=keras.initializers.Ones()))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('odd_even_check',monitor='val_acc',verbose=1, save_best_only=True, mode='max')
model.fit(train_x, train_y, epochs=10, batch_size=1000,verbose=1,callbacks=[checkpoint],validation_data=[test_x,test_y])
