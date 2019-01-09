from keras.models import Sequential
from keras.layers import  Convolution2D
from keras.layers import  MaxPooling2D
from keras.layers import  Flatten
from keras.layers import  Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
# First Layer
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation="relu"))
#Feature mapping
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Second Layer Feature mapping
classifier.add(Convolution2D(32,3,3,activation="relu"))
#Feature mapping
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Convert to single array
classifier.add(Flatten())

classifier.add(Dense(output_dim =128, activation= 'relu'))
classifier.add(Dense(output_dim =3, activation= 'softmax')) #softmax default
classifier.compile(optimizer="adam", loss = 'categorical_crossentropy', metrics= ["accuracy"])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

traning_set = train_datagen.flow_from_directory(
        'nic/train_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'nic/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        traning_set,
        steps_per_epoch=25,
        epochs=5,
        validation_data=test_set,
        validation_steps=7)


classifier.save('idt.h5')
