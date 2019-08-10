from src import train, predict
from load_data import load_dataset
import numpy as np
import os

is_train = False
save_name = 'mitTest'
run_name = 'test1'
#file = 'RML2016.10a_dict.pkl'
file = 'mitbih_test.csv'

if not os.path.exists('data/{}_in.npy'.format(save_name)):
    data, labels = load_dataset(file, dtype='csv', save_data=save_name)

data = np.load('data/radio_in.npy') if is_train else np.load('data/{}_in.npy'.format(save_name))
labels = np.load('data/radio_lab.npy') if is_train else np.load('data/{}_lab.npy'.format(save_name))
cat_names = ["N", "S", "V", "F", "Q"]
print(np.shape(data))
print(labels[:10])
print(np.sum(labels, 0))

sess = train.start_tf_sess()
if is_train:
    train.train(sess=sess, data=data, labels=labels,
                learning_rate=.000001, run_name=run_name, steps=5000, batch_size=10, accumulate=0, print_each=25)
else:
    predictions = predict.predict(sess, data=data, run_name=run_name, batch_size=100, num_categories=len(labels[0]),
                                  category_names=cat_names)
    predict.prediction_accuracy(predictions, labels, True)
