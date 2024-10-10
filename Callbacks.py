# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:19:36 2020

@author: Bijak Rabbani
"""

from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import math 

def step_decay(epoch, initial_lr=0.001, drop=0.25, epochs_drop=25):
	lrate = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


class Metrics(keras.callbacks.Callback):


    def __init__(self, model, val_data, validation_steps = 15, batch_size = 32, save_result=False):
        super().__init__()
        self.model = model
        self.validation_data = val_data
        self.batch_size = batch_size
        self.validation_steps = validation_steps
        self.save_result = save_result
        self.counter = 1

    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        bs = self.batch_size 
        y_pred = np.array([])
        y_val = np.array([])
        w_size = 0
        
        for idx in range(int(self.validation_steps)):
            l = self.validation_data.__getitem__(idx)
            l_len = bs
            y_val = np.append(y_val, l[1].argmax(axis=1))
            y_pred = np.append(y_pred, np.array(self.model.predict(l[0])).argmax(axis=1))                  
            w_size = w_size + l_len

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)
        print(f" val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            if self.save_result:
                self.model.save('result/' + self.model.model_name + '/'+'model.h5')

        self.counter += 1
        return 
    
