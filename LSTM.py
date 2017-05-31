from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tensorflow as tf
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

################################
# Load data from file
def loaddata(csvname,scaler,traintestfrac):

    # Read csv
    dataframe = pd.read_csv(csvname, usecols=[2],engine='python',skiprows=1,skipfooter=30,converters={'col':np.float})
    dataframepredicted = pd.read_csv(csvname, usecols=[1], engine='python',skiprows=1)   

    # Massage data, interpolate missing entries
    dataframe.fillna(np.nan)
    dataframe.interpolate()
    dataset = dataframe.values    
    dataset = dataset.astype('float32')
    
    # Scale data
    dataset = scaler.fit_transform(dataset)
        
    # Split data for training and testing
    len_train = int(len(dataset)*traintestfrac)
    len_test  = int(len(dataset)-len_train)

    print(len_train,len_test)    
    train = dataset[0:len_train,:]
    test  = dataset[len_train:len(dataset),:]
    return dataset,train,test

################################
# Format data for ( time-n) inputs
def create_dataset(dataset, timeback):
    dataX, dataY = [], []
    for i in range(len(dataset)-timeback-1):
        a = dataset[i:(i+timeback), 0]
        dataX.append(a)
        dataY.append(dataset[i + timeback, 0])
    return np.array(dataX), np.array(dataY)

################################
# Define NN architecture with Keras (magic)
def create_model(neurons,batchsize,timeback):
    # Create sequential model
    model = Sequential()

    # Architecture is a LSTM with 2-hidden layes
    model.add(LSTM(neurons, batch_input_shape=(batchsize, timeback,1), stateful=True, return_sequences=True))
    model.add(LSTM(neurons, batch_input_shape=(batchsize, timeback,1), stateful=True, return_sequences=True))
    model.add(LSTM(neurons, batch_input_shape=(batchsize, timeback,1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')    
    return model

################################
# Plot
def makeplot(scaler,dataset,trainPredict,testPredict,timeback):
    # Shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[timeback:len(trainPredict)+timeback, :] = trainPredict

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(timeback*2)+1:len(dataset)-1, :] = testPredict

    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    
################################
# Main
################################
if __name__ == "__main__":

    # Parameters:    
    epochs        = 100
    neurons       = 20
    timeback      = 5 
    batchsize     = 6
    traintestfrac = 0.75
    rndseed       = 1234

    np.random.seed(rndseed)
    
    # Create sklearn Scaler (to "normalize" data and make the LSTM happy)
    scaler  = MinMaxScaler(feature_range=(0,1))

    # Load and format data-inputs:
    dataset,train,test = loaddata("data/test.csv",scaler,traintestfrac)

    # Reshape into X=t and Y=t+1 (timeback=1)
    trainX, trainY = create_dataset(train, timeback)
    testX, testY = create_dataset(test, timeback)

    # Reshape input to be [samples, time-steps, features] -> keras wants numpy arrays this way
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1] ,1))
    testX  = np.reshape(testX , (testX.shape[0] , testX.shape[1]  ,1))
 	
    # Create Model
    model =  create_model(neurons,batchsize,timeback)
    
    # Fit Model, 
    for i in range(epochs):
        print("Epoch ",i)
        model.fit(trainX, trainY, epochs=1, batch_size=batchsize, verbose=1, shuffle=False)
        model.reset_states()

    # Predict sets 
    trainPredict = model.predict(trainX,batch_size=batchsize)
    testPredict   = model.predict(testX,batch_size=batchsize)
 
    # Invert-normalize predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY           = scaler.inverse_transform([trainY])
    testPredict  = scaler.inverse_transform(testPredict)
    testY            = scaler.inverse_transform([testY])

    # Calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    makeplot(scaler,dataset,trainPredict,testPredict,timeback)
    
#if we don't want batches:
#model.add(LSTM(4,input_shape=(timeback,1)))
#model.fit(trainX,trainY,epochs=100,batchsize=1,verbose=1)    
