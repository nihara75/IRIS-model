from flask import Flask,request,jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def return_predictions(model,scaler,sample_json):
  
  sep_l=sample_json['sepal_length']
  sep_w=sample_json['sepal_width']
  pep_l=sample_json['petal_length']
  pep_w=sample_json['petal_width']

  flower=[[sep_l,sep_w,pep_l,pep_w]]
  flower=scaler.transform(flower)
  classes=np.array(['setosa', 'versicolor', 'virginica'])
  class_ind=model.predict_classes(flower)
  return classes[class_ind][0]

app=Flask(__name__)

@app.route("/")
def index():
	return '<h1>FLASK APP IS RUNNING</h1>'

flower_model=load_model("final_iris_model.h5",compile=False)
flower_scale=joblib.load("iris_scaler.pkl")

@app.route('/api/flower',methods=['POST'])
def flower_prediction():
	content=request.json
	results=return_predictions(flower_model,flower_scale,content)
	return jsonify(results)




if __name__=='__main__':
	app.run(debug=True)
