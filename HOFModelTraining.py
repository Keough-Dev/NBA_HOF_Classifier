#!/usr/bin/env python
# coding: utf-8

#CREATE MODEL USING LARGE AMOUNT OF CAREER FEATURES IGNORING ERA
#(essentially a baseline)


# read in many features that focus on career totals and that ignore era
import pickle

pickling_off = open("statsHOF.pickle","rb")
stats_complete_agg = pickle.load(pickling_off)


features = stats_complete_agg.iloc[: , [1,4,5,7,8,9,10,13,14,15,16,18,19,20,
                                        21,23,50,51,52,53,54,55,56,57,58,59,60,
                                        61,62,63,64,65,66,67,68,69,70,71,72,
                                        73,74,75,76,77,78,80,81, 88]]

# set our target to the hall of fame binary classifier

target = stats_complete_agg.iloc[: , 89]

# import scikit learn libraries that we will use to train our first two models
# (hopefully Guido van Rossum can forgive me for using imports somewhat
# sporatically in this file)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale

# scale our features so as not to skew the classification with some large and
# some small values

scaled_features = scale(features)

# get rid of nans and print out a preview of our scaled features

scaled_features = np.nan_to_num(scaled_features)

# create our model, divide data into training and testing sets using a seed for
# reproduceability, and fit the model to our data

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(scaled_features, target,
                                   test_size = 0.2, random_state=20)

logreg.fit(X_train, y_train)

predictions = logreg.predict_proba(scaled_features)

predictions = predictions[:,1]

predictions = pd.DataFrame(predictions)

stats_complete_hof = stats_complete_agg.join(predictions)

stats_complete_hof = stats_complete_hof.rename(columns = {0 : 'HOF_Prob'})

# zip our column names to our coefficient values and print them out
# so they can be analyzed and interpreted

cos = [*zip((np.array(features.columns)), (logreg.coef_).reshape(-1,1))]


# In[20]:


cos


"""
Now I'll go through a quick examination of the above results,
some things that jump out right away are the large size of the model
score and how close it is to 1.
However in our case, this number is very misleading.
The number of non hall of famers is so large
and so many players are rather obviously
not HOF candidates based on their careers,
so even with a very poor score on fringe players and actual hall of famers,
a model can produce a very large score in this case.
What we are almost more concerned with here are the actual probabilities
produced by our model. Without going into too much detail at this time,
right off the bat there are some pretty large discrepancies when applying some
domain and previous player knowledge.
Lastly, some notes on the coefficiants produced,
there is a lot of multicollinearity at this point
(it's just a baseline so it's okay) and also stats such as ABA finals MVP are
weighed much more heavily than NBA finals MVPs
(which are basically negligable right now) which seems rather off,
showing that era adjustment as well as removing correlated
features can greatly improve our model's performance
"""

# CREATE MODEL USING CAREER AVERAGES, REMOVING ERA SPECIFIC FACTORS

# create second scikit learn model using similar methods,
# only this time we will use career averages rather than career totals
# along with removing stats that are specific to certain eras

Era_Free_Features = stats_complete_agg.iloc[: , [50, 51, 52, 53, 54, 55, 56, 62,
                                                 63, 64, 65, 66, 67, 68, 82,
                                                 83, 84, 85, 86, 88]]

Era_Free_Features = Era_Free_Features.fillna(0)

Era_Free_Targets = stats_complete_agg.iloc[:, 89]

Era_Free_Targets = Era_Free_Targets.fillna(0)

# Play around with hyperparameters for this second model,
# seeing if regularization will have a big impact on our model's results
# K fold validation included this time as additional validation precaution

from sklearn.model_selection import GridSearchCV

logreg_era_free = LogisticRegression()

era_free_features_scaled = scale(Era_Free_Features)

X_era_train, X_era_test, y_era_train, y_era_test = train_test_split(
era_free_features_scaled, Era_Free_Targets, test_size = .2, random_state = 20)

tester = GridSearchCV(logreg_era_free,
                      param_grid = {'C' : [0.001, 0.1 , 1]}, cv = 5)

tester.fit(X_era_train, y_era_train)

tester.score(X_era_test, y_era_test)

# we see a slight increase in model accuracy,
# which given the small sample size of
# actual inductees that we're dealing with and mentioned before,
# this is actually a substantial increase
# (we will adress the actual sensitivity and specificity later)

tester.score(era_free_features_scaled, Era_Free_Targets)

era_free_predicts = tester.predict_proba(era_free_features_scaled)

era_free_predicts = pd.DataFrame(era_free_predicts[:, 1])

stats_complete_hof = stats_complete_hof.join(era_free_predicts)
stats_complete_hof = stats_complete_hof.rename(
                                        columns = {0 : 'era_free_hof_prob'})


# CREATE MODEL USING ERA FREE CRITERIA IN KERAS

# import all keras libraries that we'll use to create the deep
# learning version of our classifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix

# set a random seed for reproduceability

np.random.seed(20)

# create model
model = Sequential()
# Input layer (with dropout)
model.add(Dropout(0.3,input_shape=(20,)))
# First hidden layer
model.add(Dense(10,  kernel_initializer='normal',
                activation='relu',kernel_constraint=maxnorm(3)))
# Second hidden layer
model.add(Dense(8,  kernel_initializer='normal',
                activation='relu',kernel_constraint=maxnorm(3)))
# Output (target) layer
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

# compile our model using binary crossentropy as our
# loss function and adam as our optimizer
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Save model to disk
# Save model structure as json
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# save our weight checkpoints to disk as well so that they can later be used

filepath = 'weights.best.hdh5'
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# save our model history to a variable for later analysis

history=model.fit(era_free_features_scaled,Era_Free_Targets,
        validation_split=0.333,epochs=250,verbose=0,callbacks=callbacks_list)

# Evaluate the model
scores = model.evaluate(era_free_features_scaled, Era_Free_Targets)
Y_predict = model.predict(era_free_features_scaled)

# create confusion matrix and variable of prediction
# values to later be appended to our dataframe
rounded = [round(i[0]) for i in Y_predict]
y_pred = np.array(rounded,dtype='int64')
CM = confusion_matrix(Era_Free_Targets, y_pred)

# create plot to analyze the progression of our
# training and testing loss throughout our epochs of training

# import matplotlib.pyplot as plt
# accuracy_training = history.history['accuracy']
# loss = history.history['loss']
# accuracy_testing = history.history['val_accuracy']
# plt.semilogy(accuracy_training,label='accuracy - training')
# plt.semilogy(accuracy_testing,label='accuracy - testing')
# plt.xlabel('Epoch' )
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# append probability values to our dataframe
# and once again examine list of leading scorers

df_Y_predict = pd.DataFrame(Y_predict)
stats_complete_hof = stats_complete_hof.join(df_Y_predict)
stats_complete_hof = stats_complete_hof.rename(columns = {0 : 'keras_hof_prob'})

# Keras provided our best model to date and in my opinion,
# looking through the three probabilty columns,
# our most accurate set of probabilities yet

# define functions using keras predictions to easily display probabilities or final prediction for any given individual player

def probCheck(player):
    print(stats_complete_hof[stats_complete_hof['Player']\
     == player]['keras_hof_prob'].values)

def predictCheck(player):
    if y_pred[stats_complete_hof[stats_complete_hof['Player']\
     == player].index] == 1:
        print("Yes")
    else:
        print("No")

"""
Ultimately while all models were fairly accurate given certain data limitations,
Keras performed most effectively
scikit learn options could still be valuable
given potential time or computing restraints.
Also all 3 models undoubtedly have more hyper parameter tuning to be done,
but again the time vs benefit dillema must be considered
(also seeing as Keras performed most effectively I did not go back and calculate
confusion matrices for other models but could have).
After considering all edge cases outside the scope of the model,
it performed nearly perfect on accuracy scores.
Aside from all of this, the biggest success of this project may be the robust
dataframe that is now available for future analyses.
"""
