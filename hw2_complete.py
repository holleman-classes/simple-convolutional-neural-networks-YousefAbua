### Add lines to import modules as needed
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras import Input, layers
from keras import datasets

from PIL import Image
## 
 
def build_model1():
    model = tf.keras.Sequential([
        Input(shape=(32, 32, 3)),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),    
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(), 
        layers.MaxPooling2D(pool_size=(4, 4), strides=(4,4)), 
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='relu'), 
    ])
    return model

def build_model2():
    model = tf.keras.Sequential([
        Input(shape=(32, 32, 3)),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),    
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        layers.BatchNormalization(), 
        layers.MaxPooling2D(pool_size=(4, 4), strides=(4,4)), 
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='relu'), 
    ])
    return model

def build_model3():
    ## This one should use the functional API so you can create the residual connections
    input_layer = Input(shape=(32,32,3))

    conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(0.25)(conv1)
    
    conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(0.25)(conv2)

    conv3 = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    
    skip1 = layers.Conv2D(128, kernel_size=(1,1), strides=(4,4))(conv1)
    skip1 = layers.add([conv3, skip1])
    skip1 = layers.Dropout(0.25)(skip1)

    conv4 = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(skip1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(0.25)(conv4)

    conv5 = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(conv4)
    conv5 = layers.BatchNormalization()(conv5)

    skip2 = layers.add([conv5, skip1])
    skip2 = layers.Dropout(0.25)(skip2)

    conv6 = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(skip2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Dropout(0.25)(conv6)

    conv7 = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(conv6)
    conv7 = layers.BatchNormalization()(conv7)

    skip3 = layers.add([conv7, skip2])
    skip3 = layers.Dropout(0.25)(skip3)

    pooling = layers.MaxPool2D(pool_size=(4,4), strides=(4,4))(skip3)
    flatten = layers.Flatten()(pooling)
    dense = layers.Dense(128, activation='relu')(flatten)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dense(10, activation='relu')(dense)

    model = keras.Model(inputs=input_layer, outputs=dense)
    return model

def build_model50k():
    model = tf.keras.Sequential([
        Input(shape=(32,32,3)),
        layers.Conv2D(16, kernel_size=(3,3), activation='relu', strides=(2,2)),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', strides=(2,2)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

    ########################################
    ##### Setup Cifar10 Dataset
    (train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.cifar10.load_data()
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    train_labels = train_labels.squeeze()
    val_labels = val_labels.squeeze()

    train_images = train_images / 255.0
    val_images  = val_images  / 255.0
    ########################################
    ##### Process Dog Image
    #image_path = 'test_image_dog.jpg'
    #image = Image.open(image_path)
    #image = image.resize((32,32))
    #image = np.array(image) / 255.0
    ########################################

    ##########
    ## Build and train model 1 (Conv2D)
    model1 = build_model1()
    model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model1.summary()
    #model1.fit(train_images, train_labels, epochs=1, batch_size=32)
    
    #predict = model1.predict(np.expand_dims(image, axis=0))
    #predict_class_index = np.argmax(predict[0])
    #predict_class_name = class_names[predict_class_index]
    #print(predict_class_name)

    ##########
    ## Build and train model 2 (DS Convolutions)
    model2 = build_model2()
    model2.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    model2.summary()
    #model2.fit(train_images, train_labels, epochs=1, batch_size=32)
    
    model3 = build_model3()
    model3.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    model3.summary()
    #model3.fit(train_images, train_labels, epochs=1, batch_size=32)

    model50k = build_model50k()
    model50k.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
    model50k.summary()
    model50k.fit(train_images, train_labels, epochs=20, batch_size=16)

    model50k.save("best_model.h5")