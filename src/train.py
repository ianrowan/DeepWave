import os
import tensorflow as tf
import numpy as np
import src.model as model
import time

def train(sess,
          data,
          labels,
          steps,
          run_name,
          batch_size=1,
          n_heads=None,
          n_layers=None,
          learning_rate=0.0001,
          print_each=1,
          save_every=1000,
          accumulate = 5,
          model_path="checkpoint/"):

    model_path = os.path.join(model_path, run_name)

    new_run = 'counter' not in os.listdir(model_path)
    if new_run: open(os.path.join(model_path, 'counter'), 'w')

    hparams = model.default_hparams()
    #Set HyperParams
    if n_layers: hparams.n_layer = n_layers
    if n_heads: hparams.n_head = n_heads

    #Spectrogram dimensions
    d_shape = np.shape(data)
    hparams.n_timestep = d_shape[1]
    hparams.n_freq = d_shape[2]
    hparams.n_cats = len(labels[0])

    #Create TF graph
    inp_specs = tf.placeholder(tf.float32, [batch_size, hparams.n_timestep, hparams.n_freq])
    logits = model.model(hparams, inp_specs,reuse=tf.AUTO_REUSE)
    #Loss tensor = Softmax cross entropy
    label_exp = tf.placeholder(tf.int8, [batch_size, hparams.n_cats])
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_exp, logits=logits['logits'])

    all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
    print("Using {} Parameter Network".format(str(len(all_vars))))

    #Train step using AdamOtimizer with Accumulating gradients
    opt = AccumulatingOptimizer(opt=tf.train.AdamOptimizer(learning_rate=learning_rate), var_list=all_vars)
    opt_reset = opt.reset()
    opt_compute = opt.compute_gradients(loss)
    opt_apply = opt.apply_gradients()

    #Create saveable graph and checkpoint + counter
    saver = tf.train.Saver(var_list=all_vars)
    sess.run(tf.global_variables_initializer())
    if new_run:
        saver.save(sess, model_path+ "/{}.ckpt".format(run_name))
    ckpt = tf.train.latest_checkpoint(model_path)
    print('Restoring checkpoint', ckpt)
    saver.restore(sess, ckpt)

    #Training SetUp
    #Get counter
    counter = 1
    counter_path = os.path.join(model_path, 'counter')
    with open(counter_path, 'r') as fp:
        counter = int(fp.read()) + 1
    counter_base = counter

    def save():
        print(
            'Saving',
            os.path.join(model_path,
                         'model-{}').format(counter-1))
        saver.save(
            sess,
            os.path.join(model_path, 'model'),
            global_step=counter-1)
    with open(counter_path, 'w') as fp:
        fp.write(str(counter-1) + '\n')

    def next_batch(num, data, lab):
        '''
        Return a total of `num` random samples and labels.
        '''
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [lab[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    try:
        while counter < (counter_base+ steps):
            if (counter - 1) % save_every == 0 and counter > 1:
                save()
            sess.run(opt_reset)
            # Get batch of specified size
            x, lab = next_batch(batch_size, data, labels)
            #Run Gradient accumulation steps
            for _ in range(accumulate):
                sess.run(opt_compute, feed_dict={inp_specs: x, label_exp: lab})
            v_loss = sess.run(opt_apply)
            avg_loss = (avg_loss[0] * 0.99 + v_loss, avg_loss[1] * 0.99 + 1.0)
            print(
                '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_loss,
                    avg=avg_loss[0] / avg_loss[1]))
            if counter % print_each == 0:
                sample = next_batch(batch_size, data, labels)
                out = sess.run(logits, feed_dict={inp_specs: sample[0]})
                acc = sum(np.argmax(np.asarray(out), axis=1) == np.argmax(sample[1], axis=1))/batch_size
                print("[Summary Step] Accuracy {}%".format(str(acc * 100)))
            counter +=1


    except KeyboardInterrupt:
        print('interrupted')
        save()

class AccumulatingOptimizer(object):
    def __init__(self, opt, var_list):
        self.opt = opt
        self.var_list = var_list
        self.accum_vars = {tv : tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                           for tv in var_list}
        self.total_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))
        self.count_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))

    def reset(self):
        updates = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars.values()]
        updates.append(self.total_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        updates.append(self.count_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def compute_gradients(self, loss):
        grads = self.opt.compute_gradients(loss, self.var_list)
        updates = [self.accum_vars[v].assign_add(g) for (g,v) in grads]
        updates.append(self.total_loss.assign_add(loss))
        updates.append(self.count_loss.assign_add(1.0))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def apply_gradients(self):
        grads = [(g,v) for (v,g) in self.accum_vars.items()]
        with tf.control_dependencies([self.opt.apply_gradients(grads)]):
            return self.total_loss / self.count_loss
