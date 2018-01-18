# -*- coding: utf-8 -*-
"""

"""

import tensorflow as tf
import os
import numpy as np
from PIL import Image

def LoadSet( path ) :
    data =[]
    labels = []
    for fname in os.listdir(path):
        pathname = os.path.join(path, fname)
        img = Image.open(pathname)
        data.append(np.asarray(img))
        labels.append(float(fname[8:-4]))
    
    data = np.array(data, dtype = np.float32)
    labels = np.array(labels, dtype = np.float32)
    return data, labels



def CNN_Model(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 128, 128, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
#  logits = tf.layers.dense(inputs=dropout, units=10)
  output = tf.layers.dense(inputs=dropout, units=1)

  predictions = {
#      # Generate predictions (for PREDICT and EVAL mode)
#      "classes": tf.argmax(input=logits, axis=1),
#      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#      # `logging_hook`.
#      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      "angle_logits": tf.squeeze(output, [1], name='logits')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
#  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
#  loss = tf.losses.softmax_cross_entropy(
#      onehot_labels=onehot_labels, logits=logits)
#  print(labels.get_shape())
#  print(output.get_shape())
  loss = tf.losses.mean_squared_error( 
          labels = labels, predictions=predictions["angle_logits"])

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer( 
            learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "RMSE": tf.metrics.root_mean_squared_error(
          labels=labels, predictions=predictions["angle_logits"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    train_data, train_labels = LoadSet( "trainset" )
    valid_data, valid_labels = LoadSet( "validset" )
    CNN = tf.estimator.Estimator(
            model_fn=CNN_Model, model_dir="tmp\CNN")
    
    train_input_fn =tf.estimator.inputs.numpy_input_fn(
            x = {"x":train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
    
    tensors_to_log = {"angle_logits":"logits"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
    
    
    CNN.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x":valid_data},
            y=valid_labels,
            num_epochs=1,
            shuffle=False)
    
    eval_results = CNN.evaluate(input_fn=eval_input_fn)
    print(eval_results)
