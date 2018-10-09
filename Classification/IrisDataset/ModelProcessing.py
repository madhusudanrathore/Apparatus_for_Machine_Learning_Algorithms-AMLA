def Run():
	import os.path
	import json

	results={}

	# if model doesn't exist,then train model
	if not os.path.exists('./Classification/IrisDataset/model_info.txt'):

		import Classification.IrisDataset.PrepareData as PD
		import numpy as NP
		from keras.utils import to_categorical
		from keras.models import Sequential
		from keras.layers import Dense, Activation	


		# GET TRAINING DATA
		PD.prepare_training_data()

		nn_training_input = NP.array(PD.training_input)
		nn_training_output = NP.expand_dims(PD.training_output,axis=1)

		# MODEL
		model = Sequential()
		model.add(Dense(100, activation='relu', input_dim=4))
		model.add(Dense(100, activation='sigmoid'))
		model.add(Dense(3, activation='softmax'))

		one_hot_labels = to_categorical(nn_training_output, num_classes=3)

		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

		nn_epochs=100
		model.fit(nn_training_input, one_hot_labels, epochs=nn_epochs)

		scores = model.evaluate(nn_training_input, one_hot_labels, verbose=0)
		
		results["accuracy"]=scores[1]*100
		results["epochs"]=nn_epochs

		# save model info
		json.dump(results, open("./Classification/IrisDataset/model_info.txt",'w'))

	else: # load data
		results=json.load(open("./Classification/IrisDataset/model_info.txt"))
	
	return results