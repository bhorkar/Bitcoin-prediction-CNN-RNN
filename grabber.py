#!/usr/bin/python3

##
## run the code for about 2/3 days
##

import requests
import time
import csv
import os.path


#f_name = input("dataset name:")
f_name = 'dataset_9_12_2017.csv'
header = ["last_updated", "price_usd","24h_volume_usd","market_cap_usd","available_supply","total_supply","percent_change_1h","percent_change_24h","percent_change_7d"]
keys = header;
header.extend(['volume','vwap','sell','buy','USD'])
if not os.path.isfile(f_name):
	f = open(f_name,"w")
	writer = csv.writer(f)
	writer.writerow(header)
else:
	f = open(f_name,"a")

while True:
 try: 
  data = requests.get("https://api.coinmarketcap.com/v1/ticker/bitcoin/").json()[0]
  print data["last_updated"]
  bstamp = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/").json() 
  bkc = requests.get("https://blockchain.info/ticker").json()
  for i in keys:
    if i in data.keys():
      f.write(str(data[i])+",")
  f.write("{},{},".format(bstamp["volume"],bstamp["vwap"]))
  f.write("{},{},{}".format(bkc["USD"]["sell"],bkc["USD"]["buy"],bkc["USD"]["15m"]))
  f.write("\n")
  f.flush()
  print bstamp
  time.sleep(9*60)
 except:
   pass
