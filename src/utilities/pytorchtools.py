#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:28:07 2021

@author: spathak
"""

import numpy as np
import torch
#import mlflow

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path_to_model, early_stopping_criteria = 'loss', best_score=None, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            path_to_model: The folder in your computer where the model gets saved
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.conf_mat_train_best=np.zeros((2,2))
        self.conf_mat_test_best=np.zeros((2,2))
        self.path_to_model = path_to_model
        self.early_stopping_criteria = early_stopping_criteria

    def __call__(self, val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1, train_loss, val_auc):

        if self.early_stopping_criteria == 'loss':        
            score = -val_loss
        elif self.early_stopping_criteria == 'auc':
            score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1, train_loss, self.path_to_model)
        elif score < self.best_score:
            self.counter += 1
            self.save_checkpoint(val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1, train_loss, self.path_to_model.split('.')[0]+'_interim'+'.tar')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1, train_loss, self.path_to_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1, train_loss, path_to_model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optim_dict' : optimizer.state_dict(),
            'loss': train_loss
        }
        torch.save(state, path_to_model)
        #mlflow.log_artifact(self.path_to_model)
        #torch.save(model.state_dict(), 'checkpoint.pt')
        self.conf_mat_train_best=conf_mat_train1
        self.conf_mat_test_best=conf_mat_test1
        self.val_loss_min = val_loss

class ModelCheckpoint:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path_to_model, criteria, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            path_to_model: The folder in your computer where the model gets saved
        """
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.conf_mat_train_best=np.zeros((2,2))
        self.conf_mat_test_best=np.zeros((2,2))
        self.path_to_model=path_to_model
        self.criteria = criteria

    def __call__(self, val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1, train_loss, val_auc):

        if self.criteria == 'loss':        
            score = -val_loss
        elif self.criteria == 'auc':
            score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1, train_loss)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1, train_loss)

    def save_checkpoint(self, val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1, train_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optim_dict' : optimizer.state_dict(),
            'loss': train_loss
        }
        torch.save(state,self.path_to_model)
        #mlflow.log_artifact(self.path_to_model)
        #torch.save(model.state_dict(), 'checkpoint.pt')
        self.conf_mat_train_best=conf_mat_train1
        self.conf_mat_test_best=conf_mat_test1
        self.val_loss_min = val_loss