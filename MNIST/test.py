from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from model import Network
from config import config

#================== obtain the data ====================
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#============== construct a test dataset ===========
x_test = x_test/255.
x_test = x_test[..., tf.newaxis]
test_ds = tf.data.Dataset.from_tensor_slices( (x_test, y_test) ).shuffle(1000).batch(config['batch_size'])

#===================== test step ===========================

@tf.function
def test_step(model, inputs, labels):
    predictions = model(inputs)
    loss_val = model.loss(labels, predictions)
    model.test_loss(loss_val)
    model.test_accuracy(labels, predictions)

#====================== training ======================
template = "Epoch {}, Accuracy {}, Loss {}"

model = Network(config= config)
# load the model's parameters
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")

EPOCHS = 5
for epoch in range(EPOCHS):
    for inputs, labels in test_ds:
        test_step(model, inputs, labels)

    print(template.format(epoch+1, model.train_accuracy, model.train_loss) )
