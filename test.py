import streamlit as st
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import keras
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
st.set_page_config(page_title="Predikcija spola i godina")
st.set_option('deprecation.showfileUploaderEncoding', False)
c1, c2= st.columns(2)
def extract_face(image, required_size=(128, 128)):
  
  pixels = np.array(image.convert('RGB'))
  pixels = cv2.cvtColor(pixels,3)

  detector = MTCNN()

  results = detector.detect_faces(pixels)

  x1, y1, width, height = results[0]['box']
  x2, y2 = x1 + width, y1 + height

  face = pixels[y1:y2, x1:x2]

  image = Image.fromarray(face)
  image = image.resize(required_size)
  face_array = asarray(image)
  return face_array

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
 
def test_image(image,Model):
  img = rgb2gray(image)
  img = cv2.resize(img, (128,128), interpolation=cv2.INTER_LANCZOS4)
  img = np.array(img)
  
  img=img/255.0;
  pred_1=Model.predict(np.array([img]))
  
  age_dict = {0:'1-2', 1:'3-9',2:'10-20',3:'21-27',4:'28-45',5:'46-65',6:'65+'}
  gender_dic=['Muško','Žensko']

  age=(np.argmax(pred_1[1][0]))

  gender=int(np.round(pred_1[0][0]))
  
  c2.header('Output')
  c2.subheader('Predviđene godine: '+age_dict[age])
  c2.subheader('Predviđen spol: '+ gender_dic[gender])
  c2.image(image)
  c2.write(img.shape)

def main():
  st.markdown('<h1 style="color:white;">Predikcija spola i dobi osobe na temelju slike</h1>', unsafe_allow_html=True)
  st.markdown("Jovana Paprić, Filip Toprek")
  page_bg_img = '''
  <style>
  body {
  background-image: url("https://images.unsplash.com/photo-1557683316-973673baf926");
  background-size: cover;
  }
  </style>
  '''
  hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
  st.markdown(hide_default_format, unsafe_allow_html=True)
  st.markdown(page_bg_img, unsafe_allow_html=True)
  upload= st.file_uploader('Priložite sliku za klasifikaciju', type=['png','jpg'])
  if upload is not None:
    im= Image.open(upload)
    img= np.asarray(im)
    img= preprocess_input(img)
    img= np.expand_dims(img, 0)
    c1.header('Input Image')
    c1.image(im)
    c1.write(img.shape)

    model = tf.keras.models.load_model("model_recg.h5")
    pixels = extract_face(im)
    test_image(pixels,model)
if __name__ == '__main__':
  main()