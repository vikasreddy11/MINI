import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from tensorflow import keras
from keras import layers
from keras.models import Sequential

(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()

X_train=X_train/255.0
X_test=X_test/255.0

X_train=X_train.reshape(-1,784)
X_test=X_test.reshape(-1,784)

model=keras.Sequential([
    layers.Dense(128,activation='relu',input_shape=(784,)),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2)

test_loss,test_accuracy=model.evaluate(X_test,y_test)

print(f'Test accuracy{test_accuracy}')

predictions = model.predict(X_test[:5])
predicted_classes = predictions.argmax(axis=1)

print(predictions)