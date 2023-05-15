from flask import Flask, request,Response
from rossmann.Rossmann import Rossmann
import pickle
import pandas as pd


#loading model
model = pickle.load(open('C:\\Users\\marco\\OneDrive\\Documents\\repos\\ds_em_producao\\rossmann-sales-prediction\\local\\model\\model_rossmann.pkl', 'rb'))

#initialize API
app = Flask(__name__)


#define endpoint
@app.route('/rossmann/predict',methods=['POST'])
def rossmann_predict():

    #getting input
    test_json = request.get_json()
    
    #testing data input
    if test_json: #data ok
        
        if isinstance(test_json,dict): #only 1 line
            test_raw = pd.DataFrame(test_json,index=[0])
        
        else: #many 
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
    else: #empty
        return Response({}, status=200, mimetype='application/json') 
    
    
    #instantiate class    
    pipeline = Rossmann()
    
    #data cleaning
    df1 = pipeline.data_cleaning(test_raw)

    #feature engineering
    df2 = pipeline.feature_engineering(df1)
    
    #data preparation
    df3 = pipeline.data_preparation(df2)
    
    #predict
    df_response = pipeline.get_prediction(model, test_raw, df3)
    
    return df_response


if __name__ == '__main__':
    app.run('192.168.1.201',debug=True) #running in local host
