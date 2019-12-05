import os
import tensorflow as tf
import numpy as np
import src.model as model
import time
import json
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
def predict(sess,
            data,
            run_name,
            batch_size,
            num_categories,
            category_names,
            model_path="checkpoint/"):

    model_path = os.path.join(model_path, run_name)

    # Load Hyperparams from model
    hparams = model.default_hparams()
    if os.path.exists(model_path+"/hparams.json"):
        with open(os.path.join(model_path, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

    d_shape = np.shape(data)
    print("Precicting for data: " + str(d_shape))
    hparams.n_timestep = d_shape[1]
    hparams.n_freq = d_shape[2]
    hparams.n_cat = num_categories

    # Create TF graph
    inp_specs = tf.placeholder(tf.float32, [batch_size, hparams.n_timestep, hparams.n_freq])
    prediction = model.model(hparams, inp_specs)

    # Get Model vars
    all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
    saver = tf.train.Saver(var_list=all_vars)
    sess.run(tf.global_variables_initializer())
    ckpt= tf.train.latest_checkpoint(model_path)
    saver.restore(sess, ckpt)

    predictions = np.zeros((len(data), num_categories))
    num_batches = int(np.ceil(len(data)/batch_size))

    for i in tqdm(range(num_batches)):
        c = batch_size

        if i * batch_size + c > len(data):
            add = (i * batch_size + c) - len(data)
            pred = sess.run(prediction,
                            feed_dict={inp_specs: np.concatenate((data[i*batch_size:], np.zeros((add, hparams.n_timestep, hparams.n_freq)))),
                                       "model/drop:0": 1.0})['logits']
            predictions[i*batch_size:] = pred[:-add]
        else:

            predictions[i*batch_size: i*batch_size+c] =\
                sess.run(prediction, feed_dict={inp_specs: data[i*batch_size: i*batch_size+batch_size],
                                                "model/drop:0": 1.0})['logits']

    cats = np.argmax(predictions, axis=1)

    return {"raw": predictions,
            "category": cats,
            "predictName": ["N", "S", "V", "F", "Q"],
            "names": category_names}


def prediction_accuracy(predictions, labels, show_matrix=False):
    # Calculate accuracy -> sum(pred == labels)/total
    label_cats = np.argmax(labels, axis=1)

    accuracy = sum(predictions["category"] == label_cats)/len(labels)
    print(accuracy)
    print("*********************************\n"
          "                         {}\n"
          "Prediction distribution: {}\n"
          "Actual Distribution:     {}".format(str(set(list(predictions["predictName"]))),
                                               str(np.sum(preprocessing.LabelBinarizer().fit_transform(predictions["category"]), 0)),
                                               str(np.sum(labels, 0))))
    print("Model accuracy: {}%".format(str(accuracy * 100)))
    print("=================================")
    if show_matrix:
        _get_confusion_matrix(y_pred=predictions["category"], y_true=label_cats,
                              title="MIT-BIH Test Result Matrix",
                              target_names=predictions["names"])
        show_final_hist(predictions["category"], label_cats,  labels,
                        predictions['names'])

    return accuracy

def show_final_hist(pred, true, labels, names):
    x = np.asarray([pred, true]).transpose()
    print(x)
    print(set(list(x[:, 0])))
    plt.hist(x, bins=range(6), histtype='bar', label=["Predicted", "Actual"], rwidth=0.8)
    plt.xticks([i for i in range(6)], names)
    plt.legend(loc="upper right")
    plt.title('Predicted vs Actual Labels')
    plt.xlabel('Category\n'
               "                         {}\n"
               "Prediction distribution: {}\n"
               "Actual Distribution:     {}".format(str(names),
                                                    str(np.sum(preprocessing.LabelBinarizer().fit_transform(
                                                       pred), 0)),
                                                    str(np.sum(labels, 0))))
    plt.show()


def _get_confusion_matrix(y_pred, y_true,
                          target_names,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize=True):
    cm = metrics.confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN +TP)
    sen = TP/(TP+FN)
    spe = TN/(TP+FP)

    print("SEN: {}\nSPE: {}".format(str(sen), str(spe)))
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is None:
        target_names = [str(i) for i in range(max(y_true) + 1)]
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
