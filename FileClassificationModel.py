#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os #to access file path on my computer
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score


# The source for reading files in a folder:
# “Python Get All the Files in a Directory | Loop through All CSV files in a folder | Python OS Module” 
# https://www.youtube.com/watch?v=_TFtG-lHNHI
folder_path = "Z:\AU Courses\Fall 2023\Digital Forensics & ML\Project\Project Dataset" #target folder contains thread 2 & thread3 data

#list of files, omitting Windows config file for the folder
list_of_files = [f for f in os.listdir(folder_path) if not (os.path.isfile(os.path.join(folder_path, f)) and f.lower() == "desktop.ini")]

#the following arrays are used to store info about each file once read
fileExtension = []
fileName = []
f_type = [] #since not all file sigs will automatically convert to text, will get the format label from the file extension for an initial identification
tmp_type = '' #temp variable used to convert file extension to uppercase string

################################################# Extracting Data #################################################

# This reads the files in a folder and places the file name into the list "fileName" and the extensions into the list "fileExtension"
# Source: How to get file extension in Python?
# https://www.geeksforgeeks.org/how-to-get-file-extension-in-python/

for file in list_of_files:
    split_tup = os.path.splitext(file) #split_tup splits a file's name and extension into an array where filename is index 0  and extension is index 1
    fileName.append(split_tup[0])
    fileExtension.append(split_tup[1])
    
    tmp_type = split_tup[1]
    f_type.append(tmp_type[1:].upper())


# Extracting the file signature from each file in dataset and storing in an array
#The KNNClassifier can't convert hex strings to a float, so I will convert the hex to a decimal
#to help with data type conversion. That's what this array will hold
magicNumberDecimal = [] 

for file_name in list_of_files: #going to read each file individually to store it's file sig
    file_path = os.path.join(folder_path, file_name) 
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'rb') as file:
                # Read the first 4 bytes for the file signature
                file_signature = file.read(4)
                fileSigToHex = file_signature.hex() #need to convert the bytes to hexadecimal string
                fileSigToDec = int(fileSigToHex,16) #converting hex string to decimal
                magicNumberDecimal.append(fileSigToDec)
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {str(e)}")


################################################# Extracting Data #################################################

# Creating a dictionary of the feature and target variables, not yet preprocessed completely
data = {
    'MagicNumberInDecimal' : magicNumberDecimal,
    'FileExtension': fileExtension,
    'FileType': f_type
}
################################################# Pre processing Data #################################################

# Creating a DataFrame from the dictionary, then converting to a csv file later for ease of access
df = pd.DataFrame(data)

# One-hot encode 'FileExtension' column using get_dummies. turning categorical data into numerical
encodeFileExt = pd.get_dummies(df['FileExtension'], prefix='FileExtension')

# Concatenate the one-hot encoded columns
df = pd.concat([df, encodeFileExt], axis=1)
df = df.drop(['FileExtension'], axis=1) # Dropping the original 'FileExtension' column


#Need to encode the file type column as well, will use LabelEncode() for target variable
labelEncoder = LabelEncoder()
encodeFileType = labelEncoder.fit_transform(df['FileType'])
df["EncodedFileType"] = encodeFileType
df = df.drop(['FileType'], axis=1) #drop original column for the file type

featureVarDF = df.drop(['EncodedFileType'], axis=1) #this includes the magic number in decimal and the encoded file extension
targetVarDF = df['EncodedFileType'] #this is the label encoded file type

#converting to csv file
df.to_csv('fileTypeData.csv', index=False)

df = pd.read_csv("fileTypeData.csv")


################################################# Pre processing Data #################################################

################################################# Train/Test Model #################################################
X_train, X_test, y_train, y_test = train_test_split(featureVarDF, targetVarDF, test_size=0.3, random_state = 1)

knn_classifier = KNeighborsClassifier(n_neighbors = 7)
knn_classifier.fit(X_train, y_train)

X_test = pd.DataFrame(X_test.values, columns=featureVarDF.columns)
model_prediction = knn_classifier.predict(X_test.values)

precision = precision_score(y_test, model_prediction, zero_division=0, average='weighted')
recall = recall_score(y_test, model_prediction, average='weighted')
print("Precision:", precision)
print("Recall:", recall)

accuracy = accuracy_score(y_test, model_prediction)
print("Accuracy:", accuracy)

#finding ideal value for k based on accuracy results
# the source of the following code block comes from the machine learning textbook
# page 122 of "A Hands on Introduction to Machine Learning by Chirag Shah"
results = []

for k in range(1, 40, 1):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    model_prediction = knn_classifier.predict(X_test.values)
    accuracy = np.where(model_prediction==y_test, 1, 0).sum() / (len(y_test))
    print ("k=",k," Accuracy=", accuracy)
    results.append([k, accuracy]) # Storing the k,accuracy tuple in results array

results = pd.DataFrame(results, columns=["k", "accuracy"])

plt.plot(results.k, results.accuracy)
plt.title("Value of k and corresponding classification accuracy")
plt.show()


# In[ ]:





# In[ ]:




