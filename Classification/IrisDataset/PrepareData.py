import csv

training_input = []
training_output = []

testing_input = []
testing_output = []

def prepare_training_data():
	filename = './Classification/IrisDataset/iris_training.csv'
	raw_data = open(filename, 'r')
	reader = csv.reader(raw_data)
	#CREATING TRAINING DATASET
	x=0
	temp_row = []
	
	for row in reader:
		column = []
		temp_row.append(row)
		
		column.append(float(temp_row[x][0]))
		column.append(float(temp_row[x][1]))
		column.append(float(temp_row[x][2]))
		column.append(float(temp_row[x][3]))
		training_input.append(column)
		
		var=temp_row[x][4]
		if var=="setosa":
			training_output.append(0)
		elif var=="versicolor":
			training_output.append(0.5)
		elif var=="virginica":
			training_output.append(1)
		x=x+1

def prepare_testing_data():
	filename = './Classification/IrisDataset/iris_testing.csv'
	raw_data = open(filename, 'r')
	reader = csv.reader(raw_data)
	#CREATING TRAINING DATASET
	x=0
	temp_row = []
	
	for row in reader:
		column = []
		temp_row.append(row)
		
		column.append(float(temp_row[x][0]))
		column.append(float(temp_row[x][1]))
		column.append(float(temp_row[x][2]))
		column.append(float(temp_row[x][3]))
		testing_input.append(column)
		
		var=temp_row[x][4]
		if var=="setosa":
			testing_output.append(0)
		elif var=="versicolor":
			testing_output.append(0.5)
		elif var=="virginica":
			testing_output.append(1)
		x=x+1

p1x = []
p1y = []
p2x = []
p2y = []
p3x = []
p3y = []
def original_graph_data():
	for y in xrange(0,30):
		if y>=0 and y<=9:
			p1x.append(testing_input[y][0])
			p1y.append(testing_input[y][1])
		elif y>=10 and y<=19:
			p2x.append(testing_input[y][0])
			p2y.append(testing_input[y][1])
		elif y>=20 and y<=29:
			p3x.append(testing_input[y][0])
			p3y.append(testing_input[y][1])
