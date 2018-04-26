import cv2
import tensorflow as tf

def cnn_model_fn(features, labels, mode):
  """
  https://www.tensorflow.org/tutorials/layers
  """
  image = features['image'] if isinstance(features, dict) else features
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    image = tf.Print(image, [tf.shape(image)], message='features: ', first_n=1, summarize=10)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    labels = tf.Print(labels, [tf.shape(labels)], message='labels: ', first_n=1, summarize=10)
  
  # Input Layer
  input_layer = tf.reshape(image, [-1, 28, 28, 1])
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    input_layer = tf.Print(input_layer, [tf.shape(input_layer)], message='input_layer: ', first_n=1, summarize=10)
  
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    conv1 = tf.Print(conv1, [tf.shape(conv1)], message='conv1: ', first_n=1, summarize=10)
  
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    pool1 = tf.Print(pool1, [tf.shape(pool1)], message='pool1: ', first_n=1, summarize=10)
  
  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    conv2 = tf.Print(conv2, [tf.shape(conv2)], message='conv2: ', first_n=1, summarize=10)
  
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    pool2 = tf.Print(pool2, [tf.shape(pool2)], message='pool2: ', first_n=1, summarize=10)
  
  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    pool2_flat = tf.Print(pool2_flat, [tf.shape(pool2_flat)], message='pool2_flat: ', first_n=1, summarize=10)
  
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    dense = tf.Print(dense, [tf.shape(dense)], message='dense: ', first_n=1, summarize=10)
  
  dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    dropout = tf.Print(dropout, [tf.shape(dropout)], message='dropout: ', first_n=1, summarize=10)
  
  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    logits = tf.Print(logits, [tf.shape(logits)], message='logits: ', first_n=1, summarize=10)
  
  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      export_outputs={
        'classify': tf.estimator.export.PredictOutput(predictions)
      })
  
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)
  
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
      labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
