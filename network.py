import util
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from   sys import argv,exit
from   keras.models import Sequential
from   keras.layers import Dense,Dropout,GRU,Reshape
from   keras.layers.normalization import BatchNormalization
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import datetime as dt
import matplotlib.dates as mdates
from pyslet.odata2.memds import InMemoryEntityContainer
from pyslet.odata2.server import Server
import threading 
import logging 
import pyslet.iso8601 as iso

SERVICE_PORT = 8080
SERVICE_ROOT = "http://localhost:%i/" % SERVICE_PORT
import pyslet.odata2.metadata as edmx
import logging, threading
from wsgiref.simple_server import make_server
cache_app = None                #: our Server instance
test_samples = 40
label = ['down', 'up']
def run_cache_server():
    """Starts the web server running"""
    server = make_server('', SERVICE_PORT, cache_app)
    logging.info("Starting HTTP server on port %i..." % SERVICE_PORT)
    # Respond to requests until process is killed
    server.serve_forever()


def load_metadata():
    """Loads the metadata file from the current directory."""
    doc = edmx.Document()
    with open('MemCacheSchema.xml', 'rb') as f:
        doc.read(f)
    return doc


file_name = 'dataset.csv'
net = None
wait_time = 530
window = 20
global net_type 
net_type = 'RNN'

def buildNetRNN(w_init="glorot_uniform",act="tanh"):
    global net
    print("Building RNN, net..")
    net = Sequential()
    net.add(Dense(12,kernel_initializer=w_init,input_dim=12,activation='linear'))
    net.add(Reshape((1,12)))
    net.add(BatchNormalization())
    net.add(GRU(40,kernel_initializer=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(70,kernel_initializer=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.3))
    net.add(GRU(70,kernel_initializer=w_init,activation=act,return_sequences=True))
    net.add(Dropout(0.4))
    net.add(GRU(40,kernel_initializer=w_init,activation=act,return_sequences=False))
    net.add(Dropout(0.4))
    net.add(Dense(1,kernel_initializer=w_init,activation='linear'))
    net.compile(optimizer='nadam',loss='mse')
    print("done!")

def buildNetCNN(w_init="glorot_uniform"):
    global net
    print("Building CNN net..")
    net = Sequential()
    net.add(Convolution1D(input_shape = (window, 12),
                        nb_filter=16,
                        filter_length=4,
                        border_mode='same'))
    net.add(BatchNormalization())
    net.add(LeakyReLU())
    net.add(Dropout(0.5))
    net.add(Convolution1D(nb_filter=8,
                        filter_length=4,
                        border_mode='same'))
    net.add(BatchNormalization())
    net.add(LeakyReLU())
    net.add(Dropout(0.5))
    net.add(Flatten())
    net.add(Dense(64))
    net.add(BatchNormalization())
    net.add(LeakyReLU())
    net.add(Dense(2))
    net.add(Activation('softmax'))
    opt = Nadam(lr=0.002)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)
    net.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

def  chart(real,predicted, timestamps, show=True):
    dates=[dt.datetime.fromtimestamp(float(ts)) for ts in timestamps]
    datenums=mdates.date2num(dates)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    ax=plt.gca()
    xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(datenums[0:test_samples], real[0:test_samples],color='g')
    plt.plot(datenums[0:test_samples], predicted[0:test_samples],color='r')
    if net_type == 'CNN':
	plt.ylabel('DOWN = 0, UP = 1.0')
    else:
	plt.ylabel('BTC/USB')
    plt.legend(['Real','Predicted'])
    plt.xlabel("Last 40 samples with 9 Minutes interval")
    plt.savefig("chart.png")
    if show:
	    plt.show()  

def predictFuture(m1,m2,old_pred,window_data,prev_closep,writeToFile=False):
    actual,latest_p, ctime = util.getCurrentData(label=True)
    if net_type == 'RNN':
    	actual = np.array(util.reduceCurrent(actual)).reshape(1,12)
    	pred = util.augmentValue(net.predict(actual)[0],m1,m2)
    	pred = float(int(pred[0]*100)/100)
    	if writeToFile:
        	f = open("results","a")
        	f.write("[{}] Actual:{}$ Last Prediction:{}$ Next 9m:{}$\n".format(time.strftime("%H:%M:%S"),latest_p,old_pred,pred))
        	f.close()

    	print("[{}] Actual:{}$ Last Prediction:{}$ Next 9m:{}$".format(time.strftime("%H:%M:%S"),latest_p,old_pred,pred))
    	return latest_p,pred, 0,0,ctime 
    else:
    	for i in xrange(1,len(window_data)):
            window_data[i-1][:] = window_data[i][:];
    	window_data[len(window_data)-1][:] = actual;
    	actual = np.array(window_data).reshape(1,window,12)
    	pred =  (net.predict(actual)[0])
	real =  util.convert_close_to_label(prev_closep,latest_p)
    	if writeToFile:
        	f = open("results","a")
        	f.write("[{}] Actual:{}$, Last actual value {}, Last Prediction:{}$ Next 9m:{}\n".format(time.strftime("%H:%M:%S"),latest_p,prev_closep,old_pred,np.argmax(pred)))
        	f.close()

    	print("[{}] Actual:{}$, Last actual value {},  Last Prediction:{}$ Next 9m:{}".format(time.strftime("%H:%M:%S"),latest_p, prev_closep, old_pred,label[np.argmax(pred)]))
    	return np.argmax(real), np.argmax(pred), window_data, latest_p, ctime



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Forecast btc price with deep learning.")
    parser.add_argument('-train',type=str,help="-train dataset.csv path")
    parser.add_argument('-run',type=str,help="-run dataset.csv path")
    parser.add_argument('-model',type=str,help='-model model\'s path')
    parser.add_argument('-iterations',type=int,help='-iteration number of epoches')
    parser.add_argument('-finetune',type=str,help='-finetune base-model path')
    parser.add_argument('-net',type=str,help='-RNN or CNN')
    args = parser.parse_args()

    doc = load_metadata()
    container = InMemoryEntityContainer(doc.root.DataServices['MemCacheSchema.MemCache'])
    mem_cache = doc.root.DataServices['MemCacheSchema.MemCache.KeyValuePairs']
   
    server = Server(serviceRoot=SERVICE_ROOT)
    server.set_model(doc)
    # The server is now ready to serve forever
    global cache_app
    cache_app = server
    t = threading.Thread(target=run_cache_server)
    t.setDaemon(True)
    t.start()
    logging.info("MemCache starting HTTP server on %s" % SERVICE_ROOT)

    print(args)
    m1 = 0;
    m2 = 0;

    global net_type
    net_type = args.net;
    #Assembling Net:
    file_name = args.run if args.run is not None else args.train
    if  net_type == 'RNN':
	    buildNetRNN()
    	    print("Loading data...")
    	    data,labels, actual_close, unixtime = util.loadDataRNN(file_name)
	    #print data, lables
    else:
    	    data,labels, actual_close, unixtime = util.loadDataCNN(file_name, window)
	    print "data shape",np.array(data).shape
	    buildNetCNN()
    #data loading:
    if  net_type == 'RNN':
    	data = util.reduceMatRows(data)
    	labels,m1,m2 =util.reduceVector(labels,getVal=True)
    print("{} chunk loaded!\n".format(len(labels)))
    window_data = data[-1]
    closep      = actual_close[-1]

    if args.run is not None:
        #Loading weights
        w_name = args.model
        net.load_weights(w_name)
        print("Starting main loop...")
        hip = 0
        reals,preds, times = [],[],[]

        for i in range(len(data)-test_samples,len(data)):
            x = np.array(data[i]).reshape(1,12)
            predicted = util.augmentValue(net.predict(x)[0],m1,m2)[0]
            real = util.augmentValue(labels[i],m1,m2)
            preds.append(predicted)
            reals.append(real)
	    times.append(unixtime[i]);
	
        while True:
							                    
	    with mem_cache.open() as collection:
        		for e in collection.itervalues():
		    		del collection[e['Key'].value]
	    try:
               	real,hip, window_data, closep, ctime = predictFuture(m1,m2,hip,window_data,closep,writeToFile=True)
                reals.append(real)
                preds.append(hip)
		times.append(ctime);
		with mem_cache.open() as collection:
		     for i in range(0,10):
		     	e = collection.new_entity()
		     	e.set_key(i)
		     	e['Value1'].set_from_value(int(preds[-i]))
		     	e['Value2'].set_from_value(int(reals[-i]))
			ttime = float(times[-i]);
			
			e['time'].set_from_value(iso.TimePoint.from_unix_time(ttime))
		     	collection.insert_entity(e)
	        with mem_cache.open() as collection:
        		for e in collection.itervalues():
            			print("%s: %d ( %d) %s \n" % (e['Key'].value, e['Value1'].value, e['Value2'].value, str(e['time'].value)))
                time.sleep(wait_time)
	    except KeyboardInterrupt:
                ### PLOTTING
		chart(reals[:-test_samples],preds[:-test_samples],times[:-test_samples],show=False)
                print("Chart saved!")
                s = input("Type yes to close the program: ")
                if s.lower() == "yes":break
                print("Resuming...")

        print("Closing..")

    elif args.train is not None:
        if args.finetune is not None:
            model_name = args.finetune
            net.load_weights(model_name)
            print("Basic model loaded!")
        epochs = args.iterations
        #Training dnn
        print("training...")
        el = len(data)-10     #Last ten elements are for testing
	tdata = np.array(data[:el])
	tlabels = np.array(labels[:el])

        net.fit(tdata,tlabels,epochs=epochs,batch_size=10)
        print("trained!\nSaving...")
	if net_type == 'RNN':
        	net.save_weights("model.h5")
	else:
        	net.save_weights("model_cnn.h5")

        print("saved!")

        ### Predict all over the dataset to build the chart
        reals,preds,times = [],[],[];
        for i in range(len(data)-test_samples,len(data)):
	    times.append(unixtime[i]);
	    if net_type == 'RNN':
            	x = np.array(data[i]).reshape(1,12)
            	predicted = util.augmentValue(net.predict(x)[0],m1,m2)[0]
            	real = util.augmentValue(labels[i],m1,m2)
            	preds.append(predicted)
            	reals.append(real)
	    else:
		x = np.array(data[i]).reshape(1,window,12)
		predicted = net.predict(x);
		print predicted
		real = labels[i]
		preds.append(np.argmax(predicted))
		reals.append(np.argmax(real))

        ### Predict Price the next 9m price (magic)
        real,hip, window_data, closep, ctime = predictFuture(m1,m2,0,window_data,closep,writeToFile=True)
        reals.append(real)
        preds.append(hip)
        times.append(ctime);
        ### PLOTTING
        chart(reals,preds, times)

    else :
        print("Wrong argument")
