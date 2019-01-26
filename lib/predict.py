import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from keras.models import  load_model
from joblib import dump, load
import datetime
from utils import helpers, config
appConfig = config.app

#Load saved preprocesser form dir
preprocesser = load('./saved_models/' + appConfig['preprocessorFile'])
#Load saved trained model
model = load_model('./saved_models/' + appConfig['trainedModel'])
# There was an issue runing model to fix used this ref https://github.com/keras-team/keras/issues/6462#issuecomment-451075345
model._make_predict_function()
selctedRows=100

def cleanData(jsonData):
    data = jsonData['data']
    jsonDataSet = json_normalize(data)
    jsonDataSet2 = jsonDataSet.iloc[:, :]
    # Create dataset using payload
    return jsonDataSet2

def doPredict(newUserData):
    # Transform data to one hot ecnoder
    transformed_newUserData = preprocesser.transform(newUserData)

    # Predit removing each column
    y_pred = model.predict(transformed_newUserData)
    
    # Presentage Calcution
    y_pred = y_pred[0][0]
    leave_precentage = y_pred*100

    return leave_precentage

def run(jsonData):
    # Load data
    # allData = pd.read_csv('./csv_data/thusitha4.csv')
    allData = cleanData(jsonData)
    dataset = allData.copy()

    rmColumns = appConfig['rmColumns']
    # Removing unnessary columns
    dataset.drop(rmColumns, axis=1, inplace=True)
    
    # helpers.printDataset(dataset)
    colNameForIndex = helpers.getColumnsList(dataset)
    users_x = dataset
    
    # ReArrange the data set
    allData = allData.iloc[0:selctedRows, 0:]
    #Get only last column for y
    sizeOfUsers = len(users_x)

    # get columns indexs
    categoricalColumns = helpers.getColumnsList(dataset, 'ids')
    # categoricalColumns = ['age','branch']
    col_1st = []
    col_2st = []
    col_3st = []
    col_probability = []
    leavingProbabilityArray = []
    # each user's row should be run
    for step in range(0, sizeOfUsers):
        predictedValues_valueBase = {}
        newUserData_row = users_x.loc[step:step].copy()
        # Predit using whole row
        leave_precentage_row = doPredict(newUserData_row)
        print('leave_precentage_row', leave_precentage_row)
        
        leavingProbabilityArray.append(leave_precentage_row)

        # Iterate each column and remove each column and calculate 
        # prediciton value without that column assign to seperate varilbe
        # sort that varile then pop most affected column 
        for columnName in categoricalColumns:
            # ! important this newUserData assign should be here
            # When column names changing reassign newUserData variable
            newUserData = users_x.loc[step:step].copy()
            
            # Data remove each column
            newUserData.at[step, columnName] = None

            # Predit with a null column
            leave_precentage = doPredict(newUserData)
            predictedValues_valueBase[leave_precentage] = columnName    

        mostAffectedColumns_array = [] 
        # analyze and pop most affected 3 column columns
        for key in sorted(predictedValues_valueBase.keys(), key=None, reverse=True):
            msg = ("%s - %s" % (key, predictedValues_valueBase[key]))
            mostAffectedColumns_array.append(msg)

        # print('mostAffectedColumns_array **** ', mostAffectedColumns_array, '******')
        # print('leavingProbabilityArray', leavingProbabilityArray )
        # print('col_1st------------',col_1st )
        # print('col_2st------------',col_2st )
        # print('col_3st------------',col_3st )

        # Create the dataset 
        col_probability.append(leavingProbabilityArray.pop())
        col_1st.append(mostAffectedColumns_array.pop())
        col_2st.append(mostAffectedColumns_array.pop())
        col_3st.append(mostAffectedColumns_array.pop())
        
    # Create the datasets
    col_probability_df = pd.DataFrame({'col_probability':col_probability})
    col_1st_df = pd.DataFrame({'col_1st':col_1st})
    col_2st_df = pd.DataFrame({'col_2st':col_2st})
    col_3st_df = pd.DataFrame({'col_3st':col_3st})
    
    # merge all dataset into one dataset
    columnsArray = [allData, col_probability_df, col_1st_df, col_2st_df, col_3st_df]
    concatenatedDf = pd.concat(columnsArray, axis=1, sort=False)

    # Create CSV file
    _time = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
    concatenatedDf.to_csv("./csv_out/customersLeavingPrediction" + _time + ".csv")
    
    # Prepare json response remove unnessary columns
    return concatenatedDf.iloc[:, -4:].to_json(orient='records')
