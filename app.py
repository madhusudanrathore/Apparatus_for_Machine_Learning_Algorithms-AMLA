from flask import Flask, render_template

app=Flask(__name__)

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/iris')
def iris():
	import Classification.IrisDataset.ModelProcessing as ir
	results=ir.Run()
	return render_template('iris.html', data=results)

@app.route('/rnn')
def rnn():
	import RNN.ModelProcessing as rnn
	results=rnn.Run()
	return render_template('rnn.html', data=results)

@app.route('/lstm')
def lstm():
	import LSTM.ModelProcessing as lstm
	results=lstm.Run()
	return render_template('lstm.html', data=results)

@app.route('/gru')
def gru():
	import GRU.ModelProcessing as gru
	results=gru.Run()
	return render_template('gru.html', data=results)

@app.route('/cnn')
def cnn():
	return render_template('cnn.html')

if __name__=='__main__':
	# app.run(host='0.0.0.0',debug=True)
	app.run(host="192.168.43.205",port=5010)