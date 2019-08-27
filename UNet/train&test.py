from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from model import UNet
from config import config

#================== obtain the data ====================
'''
Assuming you have inputs, and labeled outputs...
(x_train, y_train), (x_test, y_test) = 
'''
#============== construct a training dataset ===========
x_train = x_train/255.
x_train = x_train[..., tf.newaxis]
train_ds = tf.data.Dataset.from_tensor_slices( (x_train, y_train) ).shuffle(1000).batch(config['batch_size'])
test_ds = tf.data.Dataset.from_tensor_slices( (x_test, y_test) ).shuffle(1000).batch(config['batch_size'])

#================ train and test steps ==================
# train step
@tf.function
def train_step(model, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss_val = model.loss(labels, predictions)
    gradients = tape.gradient(loss_val, model.trainable_variables)
    model.optimizer.apply_gradients( zip(gradients, model.trainable_variables) )

    model.train_loss(loss_val)
    model.train_accuracy(labels, predictions)

# test step
@tf.function
def test_step(model, inputs, labels):
    predictions = model(inputs)
    loss_val = model.loss(labels, predictions)
    model.test_loss(loss_val)
    model.test_accuracy(labels, predictions)

#====================== training ======================
template = "Epoch {}, Accuracy {}, Loss {}"

# model and a checkpoint (parameter value saver)
model = UNet(config= config)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

EPOCHS = 5
for epoch in range(EPOCHS):
    for inputs, labels in train_ds:
        train_step(model, inputs, labels)
    ckpt.step.assign_add(1)
    save_path = manager.save()

    for inputs, labels in test_ds:
        test_step(model, inputs, labels)

    print(template.format(epoch+1, model.test_accuracy, model.test_loss) )
