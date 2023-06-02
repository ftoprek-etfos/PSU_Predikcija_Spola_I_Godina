import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from tqdm.notebook import tqdm
from PIL import Image
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras.utils import load_img
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input

BASE_DIR = 'E:\\Projektni\\train_data\\'

# ovdje spremamo godine, put do slike i spol osobe
image_paths = []
age_labels = []
gender_labels = []
classes = 7

# izvlacimo potrebne informacije iz imena slike slika je strukturirana na sljedeci nacin
# 29_1_2 prvi broj nam oznacuje godine osobe, drugi broj je spol 0 musko 1 zensko, treci broj je rasa osobe, ali to necemo koristiti
for filename in tqdm(os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

df = pd.DataFrame()
df['image'], df['gender'] = image_paths, gender_labels
# ova funkcija prolazi kroz listu slika, 
# otvara svaku sliku, mijenja njenu veličinu na 128x128 piksela, 
# pretvara je u crno-bijelu verziju te je zatim pretvara u NumPy niz. 
# Naposljetku, vraća se niz koji sadrži izdvojene značajke slika.
def convert(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = img.resize((128, 128), Image.LANCZOS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

X = convert(df['image'])
# normaliziramo vrijednosti
X = X/255.0

y_gender = np.array(df['gender'])
y_age = np.array(age_labels)

# definiramo ulazni oblik 
input_shape = (128, 128, 1)
inputs = Input(input_shape)
# konvolucijski sloj
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2)) (conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2)) (conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)

dropout = Dropout(0.25)(maxp_4)

flatten = Flatten() (dropout)
#dob
dense_1_a = Dense(256,activation="relu")(flatten)
dense_2_a = Dense(128,activation="relu")(dense_1_a)
dense_3_a = Dense(64,activation="relu")(dense_2_a)
dense_4_a = Dense(32,activation="relu")(dense_3_a)
dropout_2_a = Dropout(0.5) (dense_4_a)

#Za spol duboki sloj
dense_1_g = Dense(256, activation='relu') (flatten)
dropout_2_g = Dropout(0.5) (dense_1_g)

output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_2_g)
output_2 = Dense(1,activation="relu",name="age_out")(dropout_2_a)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy','mae'], optimizer='adam', metrics=['accuracy'])

# treniramo model sa splitom 80% test 10% validacija
history = model.fit(x=X, y=[y_gender, y_age], batch_size=20, epochs=10, validation_split=0.1)

# spremanje modela
model_name = 'model_recg.h5'
model.save(model_name)



# prikazujemo graf preciznosti i loss graf za spol
acc = history.history['gender_out_accuracy']
val_acc = history.history['val_gender_out_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Gender Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['gender_out_loss']
val_loss = history.history['val_gender_out_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

# takodjer isto i za godine
loss = history.history['age_out_accuracy']
val_loss = history.history['val_age_out_accuracy']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training accuracy')
plt.plot(epochs, val_loss, 'r', label='Validation accuracy')
plt.title('Age Accuracy Graph')
plt.legend()
plt.show()