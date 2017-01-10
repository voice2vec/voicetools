#!/usr/bin/env python
#coding: utf-8
from time import time

import scipy.io.wavfile as wav

import numpy as np
from numpy.lib import stride_tricks
from sklearn.metrics import roc_auc_score

import theano
import theano.tensor as T

import lasagne

#from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer, Conv1DLayer, MaxPool1DLayer, GlobalPoolLayer, get_output, get_all_params, get_all_param_values, set_all_param_values
from lasagne.layers import dimshuffle

from lasagne.nonlinearities import very_leaky_rectify, tanh
from lasagne.updates import adagrad

from librosa import load, logamplitude
from librosa.feature import melspectrogram

SOUND_SHAPE = (10, 513)





""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

""" plot spectrogram"""
def get_spectrogram(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    return ims/200.

def make_shape_10(arr):
    arr = np.vstack((arr, np.zeros((10-arr.shape[0]%10, arr.shape[-1]))))
    arr = arr.reshape((-1, 10, arr.shape[-1]))
    return np.array([arr])

def _iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(ind) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def make_speechtovec(incoming, sound_shape, num_units, **kwargs):
    """
    :param incoming: the layer feeding into this layer, or the expected input shape.
    :param sound_shape: shape of freq x time
    :param num_units: output vector dimension
    """


    input_reshape = lasagne.layers.ReshapeLayer(incoming, (-1,) + sound_shape)  # Сворачиваем все записи друг за другом
    convolution = lasagne.layers.Conv1DLayer(input_reshape, num_filters=100, filter_size=5,
                              nonlinearity=very_leaky_rectify, name="Convolutional")
    pooling = lasagne.layers.MaxPool1DLayer(convolution, 2)
    global_pooling = lasagne.layers.GlobalPoolLayer(pooling)
    dense = lasagne.layers.DenseLayer(global_pooling, num_units=300, name="Dense")
    output_dense = lasagne.layers.DenseLayer(dense, num_units=num_units,
                                             nonlinearity=lasagne.nonlinearities.linear, name='output')
    all_vectors_output = lasagne.layers.ReshapeLayer(output_dense, (-1, 3, num_units))

    return all_vectors_output, output_dense



class Network(object):
    """docstring for Network."""
    def __init__(self,
                    sim_voice=True,
                    load_similar_weights=True,
                    load_vect=True,
                    vectorizer_weights_file_name="vectorizer_weights.npy",
                    similar_weights_file_name="simvoice_weights.npy",
                    out_vec_size=300):
        super(Network, self).__init__()
        self.sim_voice = sim_voice
        input_triplets = T.tensor4("Triplets input", dtype="float32")
        # input_two = T.tensor4("People input", dtype="float32")

        self._triplets_input = lasagne.layers.InputLayer((None, None) + SOUND_SHAPE, input_var=input_triplets)
        dism = lasagne.layers.dimshuffle(self._triplets_input,[0,1,3,2])
        _ ,self._vectorizer_l= make_speechtovec(dism, SOUND_SHAPE[::-1], out_vec_size)

        if sim_voice:
            sim_voice_input_var = T.tensor3("Similar input", dtype="float32")
            self._similar_inp = lasagne.layers.InputLayer((None, 2, 300), input_var=sim_voice_input_var)
            nn = lasagne.layers.batch_norm(self._similar_inp)
            conv_layer = lasagne.layers.Conv1DLayer(nn, 300, 2)
            nn = lasagne.layers.batch_norm(conv_layer)
            dense0 = lasagne.layers.DenseLayer(nn, 150)
            nn = lasagne.layers.batch_norm(dense0)
            dense0 = lasagne.layers.DenseLayer(nn, 50)
            nn = lasagne.layers.batch_norm(dense0)
            self._output = lasagne.layers.DenseLayer(nn, 1, nonlinearity=lasagne.nonlinearities.sigmoid)


            VvsV = lasagne.layers.get_output(self._output) #voice versus voice
            self._VvsVfun = theano.function([self._similar_inp.input_var],VvsV ,allow_input_downcast=True)

            self._predict_similar = VvsV

        if load_similar_weights:
            param = np.load(similar_weights_file_name)
            lasagne.layers.set_all_param_values(self._output, param)

        if load_vect:
            param = np.load(vectorizer_weights_file_name)
            lasagne.layers.set_all_param_values(self._vectorizer_l, param)


        vector_pred = lasagne.layers.get_output(self._vectorizer_l)
        self._vectorizer_fun = theano.function([self._triplets_input.input_var],vector_pred ,allow_input_downcast=True)

    def _make_good_data(voice_array=None, path=None):
        if path != None:
            spec = get_spectrogram(path)

            if spec.shape[-2]//10 > 0:
                voice_array = make_shape_10(spec)

            else:
                voice_array = np.array([[spec]])
        else:
            voice_array = make_shape_10(voice_array)

        return voice_array

    def vectorizer(self, voice_array = None, path = None):
        """voice_array: matrix [time, frequency]
        len(frequency) = 513"""
        assert voice_array is not None or path is not None
        if path != None:
            spec = get_spectrogram(path)

            if spec.shape[-2]//10 > 0:
                voice_array = make_shape_10(spec)

            else:
                voice_array = np.array([[spec]])
        else:
            voice_array = make_shape_10(voice_array)
        return self._vectorizer_fun(voice_array)

    def simvoice(self, voice_array=None, paths=None, vectors=None):
        """voice_array: 3 dimensional tensor [sample0, time, frequency]
            len(sample0) = 2
            len(frequency) = 513
        paths: list
            len(paths) = 2
        vectors: matrix, shape = 2, 300"""

        assert (voice_array is not None or paths is not None or vectors is not None)
        if not self.simvoice:
            raise AttributeError("When I was initialized, you have said 'simvoice=False'\nPlese, do something with this")

        if paths:
            vec1 = np.mean((self.vectorizer(path=paths[0])), axis=0)
            vec2 = np.mean((self.vectorizer(path=paths[1])), axis=0)
        elif voice_array:
            vec1 = np.mean((self.vectorizer(voice_array=voice_array[0])), axis=0)
            vec2 = np.mean((self.vectorizer(voice_array=voice_array[1])), axis=0)
        else:
            return self._VvsVfun([vectors])
        return self._VvsVfun([[vec1, vec2]])
    #
    # def train(self, X, y, X_val=None, y_val=None, count_epoch=100,
    #             batchsize=1000, iterate_minibatches=_iterate_minibatches):
    #
    #     print("### Готовимся ###")
    #     target = T.ivector("Target")
    #     parametrs = lasagne.layers.get_all_params(self._output, trainable=True)
    #
    #     loss = lasagne.objectives.binary_crossentropy(predict, target).sum()
    #     acc = lasagne.objectives.binary_accuracy(predict, target).mean()
    #     updates = adagrad(loss, parametrs, learning_rate=0.01)
    #
    #     train = theano.function([self._triplets_input.input_var, target],
    #                                 updates=updates,
    #                                 allow_input_downcast=True)
    #
    #     acc_fun = theano.function([self._triplets_input.input_var, target],
    #                                 [self.predict, loss],
    #                                 allow_input_downcast=True)
    #
    #     if not(X_val and y_val):
    #         X_val, y_val = X[:1000], y[:1000]
    #         X, y = X[1000:], y[1000:]
    #
    #
    #     print("### Тренируем ###")
    #     for epoch in range(count_epoch):
    #         st = time()
    #         for i, batch in enumerate(iterate_minibatches(X, y, batchsize)):
    #             x_tr, y_tr = batch
    #             train(x_tr, y_tr)
    #
    #         pred, los = self.acc_fun(X_val, y_val)
    #         print("Epoch: {}\tTime: {:.2f} min:", (time()-st)/60.)
    #         print("\tRocAucScore: ", roc_auc_score(y_val, pred))
    #         print("Loss: ", los)
    #         print("Output example: ", pred[:10])
    #     print("### END ###")
