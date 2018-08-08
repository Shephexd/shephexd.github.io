---
layout: post
title: Tensorflow - How to use
published: True
categories: 
- Tensorflow
tags:
- Machine learning
- Tensorflow
- Linear algebra
typora-root-url: /Users/shephexd/Documents/github/pages/
---



One of the most popular library for `Deep learning` and `Machine learning` is  `Tensorflow` by google.

This libray support **GPU computing** and **powerful visualization** for your training process.



In this post, i will introduce the below topics about `tensorflow`.



- Tensorflow concept
- Computation
- Optmize model
- Visualize model



<!--more-->





## Tensorflow Alogorithm process



1. Prepare dataset
2. Data transform and normalization
3. split dataset into train, test, valid set
4. setting hyperparameters
5. define model structure
6. define loss function
7. init model and train
8. evaluate model
9. fitting hyperparameter
10. apply model for your problem





## Tensorflow concept

Tensorflow means the computed tensors by follwing flows.

> In mathematics, **tensors** are geometric objects that describe 	linear relations](https://en.wikipedia.org/wiki/Linear_relation) between geometric vectors, scalars, and other tensors.



Your calculation will be processed sequentially as you defined tensors and arithmetic opertations.



### Tensor

1. Fixed tensor

   ```python
   row_dim, col_dim = 10, 4
   
   # Tensor filled zeros
   zero_tsr = tf.zeros([row_dim, col_dim])
   
   # Tensor filled ones  
   ones_tsr = tf.ones([row_dim, col_dim])
   
   # Tensor filled fixed number  
   filled_tsr = tf.fill([row_dim, col_dim], 42)
   
   #Tensor with constant numbers  
   constant_tsr = tf.constant([1, 2, 3])
   ```

   

2. similar shape of tensor

   ```python
   zeros_similar = tf.zeros_like(constant_tsr)
   ones_similar = tf.ones_like(constant_tsr)
   ```

   

3. Permutation tensor

   ```python
   linear_tsr = tf.linspace(start=0., stop=1., num=3)  # [0.0, 0.5, 1.0]
   integer_seq_tsr = tf.range(start=6., limit=15., delta=3)  # [6, 9, 12]
   ```

   

4. Random tensor

   1. Tensor from distribution

      ```python
      # uniform distribution
      randunit_tsr = tf.random_uniform([row_dim, col_dim], minval=0, maxval=1) # minval <= x < maxval
      
      # normal distribution
      randnorm_tsr = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
      
      # normal distribution where values in |2 * stddev|
      runcnorm_tsr = tf.truncated_normal([row_dim, col_dim], mean=0.0, stddev=1.0) # 2 * stddev < x < -2 * stddev
      ```

      

   2. Shuffled tensor

      ```python
      # Randomly shuffle
      shuffled_output = tf.random_shuffle(input_tensor)
      
      # Randomly crop to crop size
      cropped_output = tf.random_crop(input_tensor, crop_size)
      ```

      

numpy array, list can be converted to tensor with the function, `convert_to_tensor()`



### Placeholder and variable

In tensorflow, you must know the differences between `placeholder` and `variable`. 

- `placeholder` is an input object for specific type and data.
- `variable` is included in model.



#### Use Variable

`````````python
my_var = tf.Variable(tf.zeros([2, 3]))
sess = tf.Session()

# helper function to initialize all variables
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
`````````



#### Use Placeholder with variable

```python
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[2, 2])
y = tf.identity(x) # x = y (identity opertaion)
x_vals = np.random.rand(2, 2)
sess.run(y, feed_dict={x: x_vals})

# It occurs error
# sess.run(y, feed_dict={x: x_vals})
```



> Variable must be initialized first before use.
>
> All variables can be initialized easily with  helper `global_variables_initializer()`.





## Computation

In tensorflow, the compuation process is followed by computing graph. Computing process follows the connected graphs.

When you want to get result from your `Tensor`, you have to run `sess` with `Tensor`.



### Matrix computation

The way to use matrix computation for tensors is similar to `numpy`. The main difference is `lazy computing`. You can get the result after running session not defining the computations.



```python
import tensorflow as tf
import numpy as np


# Matrix computation
identity_matrix = tf.diag([1.0, 1.0, 1.0])
A = tf.truncated_normal([2, 3])
B = tf.fill([2, 3], 5.0)
C = tf.random_uniform([3, 2])
D = tf.convert_to_tensor(np.array(
    [[1., 2., 3.],
     [-3., -7., -1.],
     [0., 5., -2.],
     ]
))
print(D)

sess = tf.Session()
print(sess.run([identity_matrix, A, B, C, D]))

''' result
Tensor("Const_1:0", shape=(3, 3), dtype=float64)
[array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]], dtype=float32), array([[ 0.05783745,  0.58782923, -0.2244595 ],
       [ 1.0929521 ,  1.6095433 , -0.55411565]], dtype=float32), array([[5., 5., 5.],
       [5., 5., 5.]], dtype=float32), array([[0.08086622, 0.62660587],
       [0.8979801 , 0.8005587 ],
       [0.28561974, 0.6266439 ]], dtype=float32), array([[ 1.,  2.,  3.],
       [-3., -7., -1.],
       [ 0.,  5., -2.]])]
'''
```



Like above code snippets, just defining the graph doesn't affect to the calculation. After you run `tf.Session` , the computing will be processed following defined tensors.



## Optimization

Using the tensor you can build your predictive model. 


The input values for model are from `feed_dict` to `placeholder`.

The `Variables` can be fitted to minimize the `loss function` by  `Optimizer`.



Then when you run `optimizer.minimize(loss)` the variable can be updated iteratively.



### Build model example

```python
true_slope = 2.0

batch_size = 50
generations = 100


x_data = np.arange(1000)/10
y_data = x_data * true_slope + np.random.normal(loc=0.0, scale=25, size=1000)

data_size = len(x_data)

train_ix = np.random.choice(data_size, size=int(data_size * 0.8), replace=False)
test_ix = np.setdiff1d(np.arange(1000), train_ix)

x_data_train, y_data_train = x_data[train_ix], y_data[train_ix]
x_data_test, y_data_test = x_data[test_ix], x_data[test_ix]

x_graph_input = tf.placeholder(tf.float32, [None])
y_graph_input = tf.placeholder(tf.float32, [None])

m = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='Slope')
output = tf.multiply(m, x_graph_input, name='Batch_multiplication')

residuals = output - y_graph_input
l1_loss = tf.reduce_mean(tf.abs(residuals), name='L1_loss')

my_optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = my_optim.minimize(l1_loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(generations):
    batch_indicies = np.random.choice(len(x_data_train), size=batch_size)
    x_batch, y_batch = x_data_train[batch_indicies], y_data_train[batch_indicies]
    _, train_loss, summary = sess.run([train_step, l1_loss, summary_op],
                                      feed_dict={
                                          x_graph_input: x_batch,
                                          y_graph_input: y_batch
                                      })

    test_loss, test_resids = sess.run([l1_loss, residuals],
                                      feed_dict={
                                          x_graph_input: x_data_test,
                                          y_graph_input: y_data_test,
                                      })
    if (i+1) % 10 == 0:
        print('Generation {} of {}. Train loss: {:.3}, Test Loss: {:.3}'.format(
            i+1, generations, train_loss, test_loss
        ))
```





## Visualization with tensorboard

To visualize interesting values you need for monitoring on tensorboard, use `tf.summary` and select and defined the values after



```python
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import io


# drawing plot for model
def gen_linear_plot(x_data, y_data, slope):
    plt.figure()
    linear_prediction = x_data * slope
    plt.plot(x_data, y_data, 'b.', label='data')
    plt.plot(x_data, linear_prediction, 'r-', linewidth=3, label='predicted_label')
    plt.legend(loc='upper left')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')


# trained slope is getting closer to true_slope
true_slope = 2.0

# hyper params
batch_size = 50
generations = 100


x_data = np.arange(1000)/10
y_data = x_data * true_slope + np.random.normal(loc=0.0, scale=25, size=1000)

data_size = len(x_data)

# splitted idx for train, test
train_ix = np.random.choice(data_size, size=int(data_size * 0.8), replace=False)
test_ix = np.setdiff1d(np.arange(1000), train_ix)

x_data_train, y_data_train = x_data[train_ix], y_data[train_ix]
x_data_test, y_data_test = x_data[test_ix], x_data[test_ix]

# define graphs
# define placeholder for train input
x_graph_input = tf.placeholder(tf.float32, [None])
y_graph_input = tf.placeholder(tf.float32, [None])

# weight value
m = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='Slope')
# predicted value
output = tf.multiply(m, x_graph_input, name='Batch_multiplication')

# loss function
residuals = output - y_graph_input
l1_loss = tf.reduce_mean(tf.abs(residuals), name='L1_loss')

my_optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = my_optim.minimize(l1_loss)

with tf.name_scope('slope_estimation'):
    tf.summary.scalar('Slope_estimation', tf.squeeze(m))

with tf.name_scope('Loss_and_residuals'):
    tf.summary.histogram('Histogram_errors', l1_loss)
    tf.summary.histogram('Histogram_residuals', residuals)


dir_name = 'tensorboard'
summary_writer = tf.summary.FileWriter(logdir=dir_name, graph=tf.get_default_graph())
summary_op = tf.summary.merge_all()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(generations):
    batch_indicies = np.random.choice(len(x_data_train), size=batch_size)
    x_batch, y_batch = x_data_train[batch_indicies], y_data_train[batch_indicies]
    _, train_loss, summary = sess.run([train_step, l1_loss, summary_op],
                                      feed_dict={
                                          x_graph_input: x_batch,
                                          y_graph_input: y_batch
                                      })

    test_loss, test_resids = sess.run([l1_loss, residuals],
                                      feed_dict={
                                          x_graph_input: x_data_test,
                                          y_graph_input: y_data_test,
                                      })
    if (i+1) % 10 == 0:
        print('Generation {} of {}. Train loss: {:.3}, Test Loss: {:.3}'.format(
            i+1, generations, train_loss, test_loss
        ))

    summary_writer.add_summary(summary, i)

    if (i+1) % 20 == 0:
        slope = sess.run(m)
        plot_buf = gen_linear_plot(x_data, y_data, slope)
        image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

        image = tf.expand_dims(image, 0)

        image_summary_op = tf.summary.image("Linear_plot", image)
        image_summary = sess.run(image_summary_op)
        summary_writer.add_summary(image_summary, i)

summary_writer.close()

```





![tensorboard_sample](/assets/post_images/tensorflow/tensorboard_sample.png)

