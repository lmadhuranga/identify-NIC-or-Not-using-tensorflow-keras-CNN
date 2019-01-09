import numpy as np
from keras.models import Sequential
from keras.models import  load_model
from keras.preprocessing import image

testimage = image.load_img('nic/prediction/2.jpg',target_size=(64,64))

testimage = image.img_to_array(testimage)
testimage = np.expand_dims(testimage,0)
model = load_model('id.h5')
results =model.predict(testimage)

print (results)
