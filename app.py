from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

import os

app = Flask(__name__)

CATEGORIES = ["Alluvial Soil","Black Soil", "Clay Soil", "Red Soil"]

model = load_model('Model_cnn.h5')

model.make_predict_function()

def model_predict(image_path):
	print("predicted")
	i = image.load_img(image_path, target_size=(200,200))
	i = image.img_to_array(i)
	i = i/255
	i = i.reshape(1, 200,200,3)
	prediction = model.predict(i)
	print("Prediction:", prediction)
	pred_name = CATEGORIES[np.argmax(prediction)]
	print(pred_name)
	if pred_name == "Alluvial Soil":
		print("alluvial.html")
		return "Alluvial","alluvial.html"

	elif pred_name == "Black Soil":
		print("black.html")
		return "Black", "black.html"

	elif pred_name == "Clay Soil":
		print("clay.html")
		return "Clay","clay.html"

	elif pred_name == "Red Soil":
		print("red.html")
		return "Red","red.html"


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
	print("Entered app.py!!")
	if request.method == 'POST':
		print("Entered here")

		file = request.files['image']
		filename = file.filename
		print("@@ Input posted = ", filename)
		file_path = os.path.join('static/user uploaded', filename)
		file.save(file_path)
		print("@@ Predicting class......")
		pred, output_page = model_predict(file_path)
		return render_template(output_page, pred_output=pred,user_image=file_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)