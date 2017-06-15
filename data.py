"Script to import data from Mercado Bitcoin API"

import requests
from datetime import datetime
from pytz import utc
from pymongo import MongoClient
from apscheduler.schedulers.blocking import BlockingScheduler

client = MongoClient()
database = client['mercado_bitcoin']
collection = database['data']

def tick():
    "Import data from Mercado Bitcoin and throw it in a MongoDB collection"

    ticker = requests.get("https://www.mercadobitcoin.net/api/ticker/").json()
    depth = requests.get("https://www.mercadobitcoin.net/api/orderbook/").json()

    date = datetime.fromtimestamp(int(ticker['ticker']['date']))
    price = float(ticker['ticker']['last'])
    v_bid = sum([bid[1] for bid in depth['bids']])
    v_ask = sum([ask[1] for ask in depth['asks']])

    collection.insert({'date':date, 'price':price, 'v_bid':v_bid, 'v_ask':v_ask})
    print(date, price, v_bid, v_ask)

def main():
    "Run tick() every 60 seconds"

    scheduler = BlockingScheduler(timezone=utc)
    scheduler.add_job(tick, 'interval', seconds=60)
    try:
        scheduler.start()
    except(KeyboardInterrupt, SystemExit):
        pass

main()
