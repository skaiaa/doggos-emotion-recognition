from keras import layers
from keras import models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np
from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))

original_dataset_dir  = './data/train'
base_dir ='./split_data'

train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count, 4))
    generator = datagen.flow_from_directory(
        directory,
        target_size = (150,150),
        batch_size=batch_size,
        class_mode='categorical'
    )
    i = 0
    for input_batch, labels_batch in generator:
        features_batch = conv_base.predict(input_batch)
        features[i*batch_size : (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size >=sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 300)
validation_features, validation_labels = extract_features(validation_dir, 100)
test_features, test_labels = extract_features(test_dir, 100)

train_features = np.reshape(train_features, (300, 4*4*512))
validation_features = np.reshape(validation_features, (100, 4*4*512))
test_features = np.reshape(test_features, (100, 4*4*512))

from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))
#model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])


# train_datagen = ImageDataGenerator(
#      rescale=1./255,
#      rotation_range=40,
#      width_shift_range=0.2,
#      shear_range=0.2,
#      zoom_range=0.2,
#      horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(224,224),
#         color_mode='rgb',
#         batch_size=32,
#         class_mode='categorical'
# )
# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(224,224),
#         color_mode='rgb',
#         batch_size=32,
#         class_mode='categorical'
# )
#
# for data_batch, labels_batch in train_generator:
#     print('data batch shape: ', data_batch.shape)
#     print('labels batch shape: ', labels_batch.shape)
#     break

history = model.fit(
     train_features, train_labels,
     epochs=30,
     validation_data=(validation_features,validation_labels)
)
model.save('doggos-imagenet-1.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validating loss')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()

plt.show()
