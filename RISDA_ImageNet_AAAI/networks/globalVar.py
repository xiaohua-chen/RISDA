from typing import KeysView
import torch
import torch.nn as nn

class gloVar():
    def __init__(self):
        self.out_new=torch.zeros(1000,2048)
        self.feature_mean=torch.zeros(1000,2048)

    def set_value(self, value):
        self.out_new=value
    
    def set_feature_mean(self, feature_mean):
        self.feature_mean=feature_mean

    def get_value(self):
        try:
            return self.out_new
        except KeyErrory:
            print('false')
            exit()
    
    def get_feature_mean(self):
        try:
            return self.feature_mean
        except KeyErrory:
            print('false')
            exit()