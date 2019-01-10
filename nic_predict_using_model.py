import numpy as np
from keras.models import  load_model
from keras.preprocessing import image

# Testing data with given images
testimage = image.load_img('predict/sample1.jpg', target_size=(64, 64))

# Converts a PIL Image instance to a Numpy array. 
# https://www.tensorflow.org/versions/r1.6/api_docs/python/tf/keras/preprocessing/image/img_to_array
testimage = image.img_to_array(testimage)

# Expand the shape of an array.
# Insert a new axis that will appear at the axis position in the expanded array shape.
# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.expand_dims.html
testimage = np.expand_dims(testimage, 0)

# Train model load 
model = load_model('nicTrainedModel.h5')

# Get the result using train model
results = model.predict(testimage)

print (results)

print('End task')
