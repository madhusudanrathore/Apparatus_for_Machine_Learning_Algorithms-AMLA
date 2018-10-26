def Run():
	import os.path
	import json

	results={}

	# if model doesn't exist,then train model
	if not os.path.exists('./RNN/model_info.txt'):

		import numpy
		from pandas import read_csv
		import math
		from keras.models import Sequential
		from keras.layers import Dense, SimpleRNN, LSTM
		from sklearn.preprocessing import MinMaxScaler
		from sklearn.metrics import mean_squared_error


		# convert an array of values into a dataset matrix
		def create_dataset(dataset, nn_look_back=1):
			dataX, dataY = [], []
			for i in range(len(dataset)-nn_look_back-1):
				a = dataset[i:(i+nn_look_back), 0]
				dataX.append(a)
				dataY.append(dataset[i + nn_look_back, 0])
			return numpy.array(dataX), numpy.array(dataY)

		numpy.random.seed(7)

		# load the dataset
		dataframe = read_csv('./RNN/airline.csv', usecols=[1], engine='python', skipfooter=3)
		dataset = dataframe.values
		dataset = dataset.astype('float32')

		# normalize the dataset
		scaler = MinMaxScaler(feature_range=(0, 1))
		dataset = scaler.fit_transform(dataset)
		# split into train and test sets
		train_size = int(len(dataset) * 0.80)
		test_size = len(dataset) - train_size
		train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

		# reshape into X=t and Y=t+1
		nn_look_back = 20
		trainX, trainY = create_dataset(train, nn_look_back)
		testX, testY = create_dataset(test, nn_look_back)


		# reshape input to be [samples, time steps, features]
		#The LSTM network expects the input data (X) to be provided with a
		#specific array structure in the form of: [samples, time steps, features].
		trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
		testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))


		# create and fit the RNN network
		model = Sequential()
		model.add(SimpleRNN(50, input_shape=(nn_look_back, 1)))
		model.add(Dense(units=10))
		model.add(Dense(units=20))
		model.add(Dense(units=1))

		nn_epochs=100
		nn_batch_size=4
		model.compile(loss='mean_squared_error', optimizer='Nadam')
		model.fit(trainX, trainY, epochs=nn_epochs, batch_size=nn_batch_size, verbose=2)

		# make predictions
		trainPredict = model.predict(trainX)
		testPredict = model.predict(testX)

		# invert predictions
		trainPredict = scaler.inverse_transform(trainPredict)
		trainY = scaler.inverse_transform([trainY])
		testPredict = scaler.inverse_transform(testPredict)
		testY = scaler.inverse_transform([testY])

		# calculate root mean squared error
		trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
		testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
		
		
		results["train_score"]=trainScore
		results["test_score"]=testScore
		results["epochs"]=nn_epochs
		results["look_back"]=nn_look_back


		# save model info
		json.dump(results, open("./RNN/model_info.txt",'w'))

	else: # load data
		results=json.load(open("./RNN/model_info.txt"))
	
	return results