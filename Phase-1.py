from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from keras.layers import Conv2D,MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib import pyplot
import opendatasets as od
import tensorflow as tf
import pandas as pd
import numpy as np
import os
od.download("https://www.kaggle.com/datasets/danish003/face-images-of-acute-stroke-and-non-acute-stroke")

def create_dataframe(root_dir):
    data = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                data.append({'file_path': file_path, 'class': class_name})
    return pd.DataFrame(data)

root_path = 'face-images-of-acute-stroke-and-non-acute-stroke/main'
dataset = create_dataframe(root_path)

train_df, test_df = train_test_split(dataset, test_size=0.2, stratify=dataset['class'], random_state=42)
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30
)
testing_data_generator = ImageDataGenerator(rescale=1./255)
training_data = training_data_generator.flow_from_dataframe(
    train_df,
    x_col='file_path',
    y_col='class',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)
testing_data = testing_data_generator.flow_from_dataframe(
    test_df,
    x_col='file_path',
    y_col='class',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)
model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(training_data, epochs=10, validation_data=testing_data)

loss, accuracy = model.evaluate(testing_data)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


def predict_image(image_path):
  img = image.load_img(image_path, target_size=(128, 128))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0
  prediction = model.predict(img_array)
  class_label = 'acute stroke' if prediction[0] > 0.5 else 'non-acute stroke'
  return class_label

image_path = '/content/face-images-of-acute-stroke-and-non-acute-stroke/main/stroke_data/aug_0_1006.jpg'
predicted_class, confidence = predict_image(image_path)
print(f'Predicted Class: {predicted_class}')
