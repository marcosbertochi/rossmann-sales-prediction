import pandas as pd
import requests
import json
from flask import Flask,request,Response
import os

#constants
#TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TOKEN = '5648527712:AAFmY3gZ98E70mM0qPS5GfMSop7rTwP-VrA'

#info about the Bot
#https://api.telegram.org/bot5648527712:AAFmY3gZ98E70mM0qPS5GfMSop7rTwP-VrA/getMe

#get updates
#https://api.telegram.org/bot5648527712:AAFmY3gZ98E70mM0qPS5GfMSop7rTwP-VrA/getUpdates

#send messages
#https://api.telegram.org/bot5648527712:AAFmY3gZ98E70mM0qPS5GfMSop7rTwP-VrA/sendMessage?chat_id=1129123995&text=Hi Marcos, I'm doing good, tks!

#webhook => to link telegram api to bot
#https://api.telegram.org/bot5648527712:AAFmY3gZ98E70mM0qPS5GfMSop7rTwP-VrA/setWebhook?url=https://rossmann-sales-prediction-1y6w.onrender.com/


def send_message(chat_id, text):

	url= 'https://api.telegram.org/bot{}/'.format(TOKEN)
	url = url + 'sendMessage?chat_id={}'.format(chat_id)

	r = requests.post(url, json={'text':text})

	print('Status Code {}'.format(r.status_code))

	return None



def load_data(store_id):
	df10 = pd.read_csv('datasets/test.csv', low_memory=False) #adjusted data path from root
	df_store_raw = pd.read_csv('datasets/store.csv', low_memory=False)

	#merge test dataset + stores info
	df_test= pd.merge (df10, df_store_raw,how='left',on='Store')

	#choose store to prediction
	df_test = df_test[df_test['Store'] == store_id]

	#check if found store
	if not df_test.empty:
		#remove closed days
		df_test = df_test[df_test['Open']!=0]
		df_test = df_test[~df_test['Open'].isnull()]
		df_test = df_test.drop('Id',axis=1)

		#convert df to json to communicate to api
		data = json.dumps(df_test.to_dict(orient='records'))

	else:
		data = 'error'

	return data


def predict(data):

	#API call
	url = 'https://rossmann-sales-prediction-1y6w.onrender.com/'
	header = {'Content-type':'application/json'}

	r = requests.post(url, data=data, headers=header)
	print('Status Code {}'.format(r.status_code))

	d1 = pd.DataFrame(r.json(),columns=r.json()[0].keys())

	return d1


def parse_message(message):

	chat_id = message['message']['chat']['id']
	store_id = message['message']['text']

	store_id = store_id.replace('/','')

	try:
		store_id = int(store_id)

	except ValueError:
		store_id = 'error'

	return chat_id, store_id


#API initialize
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
	if request.method == 'POST':
		message = request.get_json()
		chat_id, store_id = parse_message(message)

		if store_id != 'error':
			#loading data
			data = load_data(store_id)

			if data != 'error':
				#prediction
				d1 = predict(data)
				#calculation
				d2 = d1[['store','prediction']].groupby('store').sum().reset_index()
				#send message
				msg = 'Store Number {} will sell R${:,.2f} in next 6 weeks'.format(d2['store'].values[0], d2['prediction'].values[0])
				send_message(chat_id,msg)
				return Response('Ok',status=200)
			else:
				send_message(chat_id, 'Store Not Found')
				return Response('Ok',status=200)

		else:
			send_message(chat_id,'Store ID wrong')
			return Response('ok', status=200)

	else:
		return '<h1> Rossmann Telegram BOT </h1>'

if __name__ == '__main__':
	port = os.environ.get('PORT', 5000)
	app.run(host='0.0.0.0', port=port, debug=True)
