from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

CATEGORIES = ["Alluvial Soil","Black Soil", "Clay Soil", "Red Soil"]

model = load_model('Model_cnn.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(200,200))
	i = image.img_to_array(i)/255
	i = i.reshape(1, 200,200,3)
	prediction = model.predict(i)
	print("Prediction:", prediction)
	pred_name = CATEGORIES[np.argmax(prediction)]
	print(pred_name)
	return pred_name


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['soil_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)