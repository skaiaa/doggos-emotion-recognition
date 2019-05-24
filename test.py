from keras import models
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
#import cv2
import tensorflow as tf

categ = ["angryboy", "goodboy", 'sleepyboy', 'smileyboy']

model=load_model('modelcool100.h5')

img_path = './cat.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
#img_tensor = preprocess_input(img_tensor)
img_tensor/= 255.

#plt.imshow(img_tensor[0])
#plt.show()

#layer_outputs = [layer.output for layer in model.layers[:8]]
#activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#activations = activation_model.predict(img_tensor)
preds = model.predict(img_tensor)
#print(categ[int(preds[0][0])])
#x = np.where(preds == np.amax(preds))[0][0]
#print(type(np.where(preds == np.amax(preds))[0][0]))
x = np.argmax(preds)
print(preds)
print(categ[x])