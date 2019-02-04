import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from keras.models import  load_model
from keras.preprocessing import image
from utils import helpers, config
appConfig = config.app
nicTypes = config.app['nicTypes']
nicIndexes = config.app['nicIndexes']

# Train model load 
model = load_model(appConfig['trainedModel'])
# There was an issue runing model to fix used this ref https://github.com/keras-team/keras/issues/6462#issuecomment-451075345
model._make_predict_function()

def preprocessing(imgName):
    imgPath = './uploads/'+imgName
    print('1 imgPath========>', imgPath)
    
    # Testing data with given images
    testImage = image.load_img(imgPath, target_size=(64, 64))
    
    # Converts a PIL Image instance to a Numpy array. 
    # https://www.tensorflow.org/versions/r1.6/api_docs/python/tf/keras/preprocessing/image/img_to_array
    imgArray = image.img_to_array(testImage)
    
    # Expand the shape of an array.
    # Insert a new axis that will appear at the axis position in the expanded array shape.
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.expand_dims.html
    imageExpanded = np.expand_dims(imgArray, 0)    

    imageExpanded /= 255
    
    return imageExpanded

def calcMatch(preds):
    final_score = preds[0][0] + preds[0][1] + preds[0][2] + preds[0][3] + preds[0][4]

    license     = preds[0][0]
    nicBack     = preds[0][1]
    nicFront    = preds[0][2]
    nicFrontNew = preds[0][3]
    other       = preds[0][4]

    presentageLic     = license/final_score * 100
    presentageNicB    = nicBack/final_score * 100
    presentageNicF    = nicFront/final_score * 100
    presentageNicFNew = nicFrontNew/final_score * 100
    presentageOther   = other/final_score * 100
    
    output = {
        nicTypes['nicFront']    : presentageNicF,
        nicTypes['nicBack']     : presentageNicB,
        nicTypes['licFront']    : presentageLic,
        nicTypes['nicFrontNew'] : presentageNicFNew,
        nicTypes['other']       : presentageOther
    }
    print('output--->', output)
    
    return output

def runPredict(imgName):
    # image preprocessed    
    preprocedImg = preprocessing(imgName)
    print('mad1')
    
    # # Get the result using train model
    preds = model.predict(preprocedImg)
    print('mad2')
    print('preds--->', preds)
    
    # # 0.licFront. 1.nicBack 2.nicFront"
    pred_classes = np.argmax(preds)
    print('mad3', pred_classes)

    print('pred_classes', pred_classes, nicIndexes[pred_classes])
    
    return {
        "nicType": nicIndexes[pred_classes],
        # "nicType": pred_classes
        
        # "presentages": calcMatch(preds)
    }
