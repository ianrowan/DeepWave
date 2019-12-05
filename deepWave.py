from src import train, predict
from load_data import load_dataset
import numpy as np
import os
import matplotlib.pyplot as plt

is_train = False
save_name = 'mitTest'
run_name = 'medNet'
#file = 'RML2016.10a_dict.pkl'
file = 'mitbih_train.csv'

if not os.path.exists('data/{}_in.npy'.format(save_name)):
    data, labels = load_dataset(file, dtype='csv', save_data=save_name)

data = np.load('data/mittrain_in.npy') if is_train else np.load('data/{}_in.npy'.format(save_name))
labels = np.load('data/mittrain_lab.npy') if is_train else np.load('data/{}_lab.npy'.format(save_name))
cat_names = ["N", "S", "V", "F", "Q"]
print(np.shape(data))
print(labels[:10])
print(np.sum(labels, 0))

sess = train.start_tf_sess()
if is_train:
    train.train(sess=sess, data=data, labels=labels,
                learning_rate=.000001, run_name=run_name,
                steps=5000, batch_size=30, accumulate=0, print_each=25, use_class_entropy=False)
else:
    predictions = predict.predict(sess, data=data, run_name=run_name, batch_size=750, num_categories=len(labels[0]),
                                  category_names=cat_names)
    predict.prediction_accuracy(predictions, labels, True)
    test_labels = np.load('data/{}_lab.npy'.format('mitTest'))
    train_labels = np.load('data/{}_lab.npy'.format('mittrain'))

    def plot_dataset_hist(test_labels, train_labels):
        x = np.asarray([np.argmax(test_labels, axis=1), np.argmax(train_labels, axis=1)]).transpose()
        print(x)
        plt.hist(x, bins=range(6), histtype='bar', stacked=True, label=["Test", "Train"], rwidth=0.4)
        plt.xticks([i for i in range(6)], cat_names)
        plt.legend(loc="upper right")
        plt.title('Test & Train Label Distribution')
        plt.xlabel('Category')
        plt.show()

    plot_dataset_hist(test_labels, train_labels)

