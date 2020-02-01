from flask import Flask, request, send_file
import base64
import numpy as np
import tensorflow
import keras
from keras.preprocessing import image
from keras.models import load_model
from flask_ngrok import run_with_ngrok
from PIL import ImageTk, Image
import fpdf
from datetime import date
import tensorflow as tf

def fetch_results(image_str,patient_name):
  with tf.name_scope("predict"):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
      model = tensorflow.keras.models.load_model("./minor-server/content/keras_model.h5")
      with open("./minor-server/content/{}.png".format(patient_name) , "wb") as image_file_recv:
        image_file_recv.write(base64.b64decode(image_str.encode('utf-8')))
      testing_image = image.load_img("./minor-server/content/{}.png".format(patient_name),target_size = (224, 224))
      testing_image = image.img_to_array(testing_image)
      testing_image = np.expand_dims(testing_image, axis = 0)

      pred = model.predict(testing_image)

      is_cancer = False
      prob = 0.0

      if pred[0][0] > pred[0][1]:
        is_cancer = True
        prob = pred[0][0]*100

      else:
        prob = pred[0][1]*100

      return {"is_cancer" : is_cancer, "prob" : prob}




app = Flask(__name__)


@app.route("/detect" , methods = ['POST'])
def detect():
  # try:
  if request.method == 'POST':
    return fetch_results(request.form.get('image_str'),request.form.get('patient_name'))
  # except:
  #   return "ERROR OCCURED", 500

@app.route("/" , methods = ['GET'])
def test():
  # try:
  if request.method == 'GET':
    return "TEST"
  # except:
  #   return "ERROR OCCURED", 500

@app.route("/getReport", methods = ['GET'])
def getReport():
  try:
    if request.method == 'GET':
      return send_file("./content/{}.pdf".format(request.args.get('patient_name')), attachment_filename = "{}_DR_Report.pdf".format(request.args.get('patient_name')))
  except:
    return "ERROR", 500


# In[ ]:

if __name__ == '__main__':
    app.config["JSON_SORT_KEYS"] = False
    app.run()







