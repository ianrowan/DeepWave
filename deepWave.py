from src import train
from load_data import load_dataset
import numpy as np

#file = 'RML2016.10a_dict.pkl'
file = 'mitbih_train.csv'
#data, labels = load_dataset(file, dtype='csv', save_data=True)
data = np.load('data/radio_in.npy')[5000:]
labels = np.load('data/radio_lab.npy')[5000:]
print(np.shape(data))
print(labels[:10])
print(np.sum(labels,0))
sess = train.start_tf_sess()

train.train(sess=sess, data=data, labels=labels, learning_rate=.000001, run_name='test1', steps=5000, batch_size=10, accumulate=0, print_each=25)
