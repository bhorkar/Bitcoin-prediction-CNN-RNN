import requests
import numpy as np
import pandas as pd
maxs =[]
mins =[]
future_step = 1;

def loadDataRNN(f_name):
    df = pd.read_csv(f_name);
    data = df.drop(['USD','last_updated'], axis = 1).as_matrix();
    label = np.transpose(df['USD'].as_matrix());
    unixtime = df['last_updated'].as_matrix(); 
    actual_close = label[future_step:] 
    return data[future_step:],label[:-future_step], actual_close, unixtime[:-future_step]  #Removing first two and last two so each X[i] tries to predict Y[i+2] (i've used i+2 and not to i+1 to force it to predict the future (O) )

def convert_close_to_label(last_close,next_close):
	if last_close > next_close:
		y_i = [1, 0] #sell
	else:
		y_i = [0, 1] #buy 
	return y_i;

def loadDataCNN(f_name, window):
    
    df = pd.read_csv(f_name);
    actual_close = [];
    data = []; 
    unixtime = [];
    label = []
    print "df shape", df.shape
    for i in range(df.shape[0]-window-10):
	last_close = df.loc[i+window,'USD']  
        next_close = df.loc[i+future_step+window,'USD']
        unixtime.append(df.loc[i+future_step+window,'last_updated'])
	y_i = convert_close_to_label(last_close,next_close);
        label.append(y_i)
        actual_close.append(last_close)
        window_i = df.loc[i+1:i+window,:].drop(['USD','last_updated'], axis = 1).as_matrix() 
	data.append(window_i)
    return data,label,actual_close, unixtime 


def reduceVector(vec,getVal=False):
    vect = []
    mx,mn = max(vec),min(vec)
    mx = mx+mn
    mn = mn-((mx-mn)*0.4)
    for x in vec:
        vect.append((x-mn)/(mx-mn))
    if not getVal:return vect
    else:return vect,mx,mn

def reduceValue(x,mx,mn):
    return (x-mn)/(mx-mn)

def augmentValue(x,mx,mn):
    return (mx-mn)*x+mn

def reduceMatRows(data):
    l = len(data[0])
    for i in range(l):
        v = []
        for t in range(len(data)):
            v.append(data[t][i])
        v,mx,mn = reduceVector(v,getVal=True)
        maxs.append(mx)
        mins.append(mn)
        for t in range(len(data)):
            data[t][i] = v[t]

    return data
def reduceCurrent(data):
    for i in range(len(data)):
        data[i] = reduceValue(data[i],maxs[i],mins[i])
    return data

def getCurrentData(label=False):
    keys = ["price_usd","24h_volume_usd","market_cap_usd","available_supply","total_supply","percent_change_1h","percent_change_24h","percent_change_7d"]
    vect = []
    data = requests.get("https://api.coinmarketcap.com/v1/ticker/bitcoin/").json()[0]
    bstamp = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/").json()
    bkc = requests.get("https://blockchain.info/ticker").json()
    for i in data.keys():
      if i in keys:
       vect.append(float(data[i]))
    vect.append(float(bstamp["volume"]))
    vect.append(float(bstamp["vwap"]))
    vect.append(float(bkc["USD"]["sell"]))
    vect.append(float(bkc["USD"]["buy"]))
    if label:return vect,float(bkc["USD"]["15m"]),data['last_updated'] 
    else : return vect

def test_data(mem_cache):
    with mem_cache.open() as collection:
        for i in range3(26):
            e = collection.new_entity()
            e.set_key(str(i))
            e['Value'].set_from_value(character(0x41 + i))
            e['Expires'].set_from_value(
                iso.TimePoint.from_unix_time(time.time() + 10 * i))
            collection.insert_entity(e)


def test_model():
    """Read and write some key value pairs"""
    doc = load_metadata()
    InMemoryEntityContainer(doc.root.DataServices['MemCacheSchema.MemCache'])
    mem_cache = doc.root.DataServices['MemCacheSchema.MemCache.KeyValuePairs']
    test_data(mem_cache)
    with mem_cache.open() as collection:
        for e in collection.itervalues():
            output("%s: %s (expires %s)\n" %
                   (e['Key'].value, e['Value'].value, str(e['Expires'].value)))
