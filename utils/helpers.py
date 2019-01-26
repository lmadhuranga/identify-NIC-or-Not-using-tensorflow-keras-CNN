import  numpy as np
import  pandas as pd
import math
import datetime

def getColumnsList(dataset, indexBy='names'):
    columnList_object = {}
    columnList_array = []
    # convert to array
    for index, columnName in enumerate(dataset.columns.values.tolist()):
        columnList_object[columnName] = index
        columnList_array.append(columnName)
    if indexBy == 'names':
        return columnList_object
    else:
        return columnList_array

# when dataset send format by selected column send back
def getColumnsIndex(dataset, selectedColumns, withNames=False):    
    columnObj = {}
    # convert to array
    for index, columnName in enumerate(dataset.columns.values.tolist()):
        columnObj[columnName] = index
    # get selected columns indexs
    returnArray = []
    for index, column in enumerate(selectedColumns):
        returnArray.append(columnObj[column])
    return returnArray


# Print Data set with its index
def printDataset(dataset):
    for index, columnName in enumerate(dataset.columns.values.tolist()):
        print(index, columnName)


# Reshape Data 
def getReshapedArray(dataFrame, selectedColumns, replacer=None):
    oneHotColumns = {}
    maxLengthColumn = {'column':'', 'count':0}
    # get unique of each onehot encoded column's
    for columnName in selectedColumns:
        columns = dataFrame[columnName].unique()
        columnCount = len(columns)
        oneHotColumns[columnName] = columns
        
        # if max found assing to variable
        if maxLengthColumn['count'] < columnCount:
            maxLengthColumn['count'] = columnCount
            maxLengthColumn['column'] = columnName


    dataArray = []
    # loop with max sized column
    for maxIndex, maxColumn in enumerate(oneHotColumns[maxLengthColumn['column']]):
        tempArray = []
        # other column loop and format
        for otherIndex, otherColumn in enumerate(selectedColumns):
            otherColumnLength = len(oneHotColumns[otherColumn])
            # Check array lenght miss match 
            if maxIndex < otherColumnLength:
                tempArray.append(oneHotColumns[otherColumn][maxIndex])
                # when other array not enough long enought fill with NULL or empoty
            else:
                tempArray.append(replacer)

        dataArray.append(tempArray)
    # format to data set with data and headers
    return pd.DataFrame(dataArray, columns=selectedColumns)


def loadData(filepath):
    return pd.read_csv(filepath, encoding='cp1252')

def loadOneHotencodedNamesCsv(df):
    columnsNames = list(df.columns.values)
    # remove first element
    del columnsNames[0]
    return columnsNames

# Load the onehotencoded CSV and get data
def loadOneHotencodedDataCsv(df):
    data = df.iloc[:,1:].values
    return data

def clearData(data):
    returnArray = []
    for x in data:
        if isinstance(x, str) | isinstance(x, int):
            returnArray.append(x)
    return returnArray

# Replace With Null if added the catgor
def cleanTestData(trainDataset, testDataset, columns, replacer=None):
    for column in columns:        
        # get training columns values values
        uniqueTrainData = trainDataset[column].unique()
        #  Remove Null values
        uniqueTrainData = clearData(uniqueTrainData)
        # run test row by row
        for index, row in enumerate(testDataset[column]):
            # check new category or not
            if row in uniqueTrainData:
                # replace with replacer
                testDataset[column][index]=replacer
            else:
                print('New category', row, 'replace with ', replacer)

    return testDataset

def getUniqueName(fileName, ext=False):
    # Create CSV file
    _time = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
    if ext:
        return (fileName + _time + ext)
    else:
        return (_time + fileName)

def allowed_file(filename, allowedExtentions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowedExtentions