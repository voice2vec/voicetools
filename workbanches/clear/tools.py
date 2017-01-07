import numpy as np
import theano.tensor as T
import theano

def normalize():
    the_list = T.vector(dtype='float64')
    normalized = the_list / the_list.max()  
    return theano.function([the_list], normalized)

normalize = normalize()

def get_energy():
    sample = T.vector(dtype='int64')
    window_size = T.scalar(dtype = 'int16')
    
    sample_sq = sample ** 2
    padded_sample = T.concatenate([T.zeros([window_size]), sample_sq])
    cumsum = T.cumsum(padded_sample)
    energy = cumsum[window_size:] - cumsum[:-window_size]
    
    return theano.function([sample, window_size], energy)

get_energy = get_energy()

def magnitude_func():
    sample = T.vector(dtype='int64')
    window_size = T.scalar(dtype = 'int16')
    
    sample_sq = abs(sample)
    padded_sample = T.concatenate([T.zeros([window_size]), sample_sq])
    cumsum = T.cumsum(padded_sample)
    magnitude = cumsum[window_size:] - cumsum[:-window_size]
    
    return theano.function([sample, window_size], magnitude)
magnitude = magnitude_func()

def sign_func(number):
    return number >= 0

def zcr_func():
    sample = T.vector(dtype='int64')
    window_size = T.scalar(dtype = 'int16')
    
    a = sample[:sample.shape[0] - 1]
    b = sample[1:]
    
    a, updates = theano.scan(fn=sign_func, sequences=a)
    b, updates = theano.scan(fn=sign_func, sequences=b)
    
    sample_sq = np.abs(a - b)
    padded_sample = T.concatenate([T.zeros([window_size]), sample_sq])
    cumsum = T.cumsum(padded_sample)
    zcr = cumsum[window_size:] - cumsum[:-window_size]
    zcr = zcr / (2 * window_size)
    return theano.function([sample, window_size], zcr)

zcr = zcr_func()


def predict_by_energy(signal, rate, window_size = 30, threshold = 0.3):
    '''
        :param signal: the array with signal
        :param rate: signal rate
        :param window_size_30: size of frame in ms
        :param threshold: threshold of prediction (from 0 to 1)
    '''
    
    window_size = int(window_size * (rate/1000))
    
    energy_answer = [True if i > threshold else False for i in normalize(get_energy(signal, window_size))]
   
    return list(zip(signal, energy_answer))

def predict_by_magnitude(signal, rate, window_size = 30, threshold = 0.3):
    '''
        :param signal: the array with signal
        :param rate: signal rate
        :param window_size_30: size of frame in ms
        :param threshold: threshold of prediction (from 0 to 1)
    '''
    
    window_size = int(window_size * (rate/1000))
    
    magnitude_answer = [True if i > threshold else False for i in normalize(magnitude(signal, window_size))]
    
    return list(zip(signal, magnitude_answer))


def predict_by_zcr(signal, rate, window_size = 30, threshold = 0.3):
    '''
        :param signal: the array with signal
        :param rate: signal rate
        :param window_size_30: size of frame in ms
        :param threshold: threshold of prediction (from 0 to 1)
    '''
    
    window_size = int(window_size * (rate/1000))
    
    zcr = [True if i > threshold else False for i in normalize(zcr(signal, window_size))]

    return list(zip(signal, zcr))

