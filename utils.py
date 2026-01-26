# import logging
import time
import os

def check_dir(path:str ,mkdir=False):
    if os.path.exists(path):
        return True
    elif mkdir:
        os.makedirs(path)
        return True
    
    return False

def get_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

class StandardScaler():
    def __init__(self, data):

        mean,std = data.float().mean(), data.float().std()

        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MinMaxScaler():
    def __init__(self, min,max):

        self._min = min
        self._max = max

    def transform(self, data):
        data = 1. * (data - self._min)/(self._max - self._min)
        data = data * 2. - 1.
        return data

    def inverse_transform(self, data):
        data = (data + 1.) / 2.
        data = 1. * data * (self._max - self._min) + self._min
        return data