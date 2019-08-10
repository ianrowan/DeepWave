import os
import pickle
import numpy as np
from scipy import signal
from sklearn.preprocessing import LabelBinarizer

#Todo: Add data download method

def load_dataset(file, dtype,
                 train_split=0.9,
                 spectrogram=True,
                 one_hot=True,
                 save_data=False):
    '''

    :param file: pickled file with data
    :param train_split: % of data for training
    :param spectrogram: Use Specrtogram data?
    :param one_hot: onehot encode data?
    :param save_data: file name to save npy
    :return: x:[example x sequence x frequency], labels: one-hot or integer like labels
    '''
    if dtype == 'pkl':
        # load dataset - format: {(Cat, SNR): [example x channels x samples], ...}
        data = pickle.load(open("data/{}".format(file), 'rb'), encoding='latin1')
        x = []
        labels = []
        for key in list(data.keys())[:50]:
            d = data[key]
            if not spectrogram:
                # append data point with I + Q channels for consolidation
                x.append(d[:, 0, :] + d[:, 1, :])

            else:
                # Get Spectrogram of all I + 0 examples in key
                spec = list(map(get_spectrogram, d[:, 0, :] + d[:, 1, :]))
                print(np.shape(spec))
                x.append(spec)
            labels.extend([key[0]] * np.shape(d)[0])

        x = np.vstack(x)

    else:
        data = np.loadtxt('data/{}'.format(file), delimiter=',')[-51000:]
        x = list(map(get_spectrogram, data[:, :-1]))
        labels = data[:, -1]
        print(set(list(labels)))
    x_shape = np.shape(x)
    #x = np.reshape(x, [x_shape[0] * x_shape[1], x_shape[2], x_shape[3]])
    print("Loaded {} examples with [{} x {}] Spectrograms".format(x_shape[0], x_shape[1], x_shape[2]))
    print("Loaded {} Labels with {} Categories".format(str(len(labels)), str(len(set(labels)))))
    labels = labels if not one_hot else LabelBinarizer().fit_transform(labels)
    if save_data:
        np.save('data/{}_in'.format(save_data), x)
        np.save('data/{}_lab'.format(save_data), labels)

    return np.asarray(x), np.asarray(labels, dtype=np.int8)


def get_spectrogram(wave):
    # get spectrogram of wave with assumed sample frequency of 100 kHz
    t, f, sxx = signal.spectrogram(wave, fs=125, nfft=1022, nperseg=36, noverlap=28)

    # Sxx = [Frequency x time] array
    # transpose to [time x frequency for time series

    sxx = sxx.transpose()
    return sxx
