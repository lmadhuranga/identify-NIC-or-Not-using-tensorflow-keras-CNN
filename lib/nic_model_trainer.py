from keras.models import Sequential
from keras.layers import  Convolution2D
from keras.layers import  MaxPooling2D
from keras.layers import  Flatten
from keras.layers import  Dense
from keras.preprocessing.image import ImageDataGenerator

# As initialize a sequential constructor
classifier = Sequential()

# now the model will take as input arrays of shape (*, 64)
# and output arrays of shape (*, 32)
# after the first layer, you don't need to specify
# the size of the input anymore:
# Initialize the first layer
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

# Feature mapping reduce the unnessary pixel and remain essential pixel only
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Initialize the Second Layer
classifier.add(Convolution2D(32, (3, 3), activation="relu"))

# Feature mapping second layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Convert multi-diamention array into single array
classifier.add(Flatten())

# Initialize the third layer
classifier.add(Dense(activation= 'relu', units =128 ))

# Initialize the fourth and last layer
# Reason to use soft max 
        # more than 2 out put
classifier.add(Dense(activation= 'sigmoid', units =3))

# Compile classifire with adam function
# categorical_crossentropy - due to use more tha one output
classifier.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

# Initialize train image geneerator object
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Import training dataset with 32 batch
traning_set = train_datagen.flow_from_directory(
    'nic/train_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Import test dataset with 32 batch
test_set = test_datagen.flow_from_directory(
    'nic/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Train model with training data set
    # @validation_data = set the validate data set   
classifier.fit_generator(
    traning_set,
    steps_per_epoch=len(traning_set),
    epochs=20,
    validation_data=test_set,
    validation_steps=len(test_set))

# Create the train model
classifier.save('nicTrainedModel.h5')
