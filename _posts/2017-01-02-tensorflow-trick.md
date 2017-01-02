---
layout: post
title: A trick for interactive tensorflow programming
---

I recently ran into an issue in tensorflow, specifically where I had trained a network 
on one set of data with images of size `[batch_size, m, m, 3]` and then I wanted to
use the learned filters on a different set of images of `[other_batch_size, n, n, 3]`.

Say I had a method

```def network(self, z):

    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[self.batch_size, 
                                    z.get_shape().as_list()[1],
                                    z.get_shape().as_list()[2], self.dim],
                                    strides=[1, d_h, d_w, 1]))
    ...
```

which we run via

```
self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3], name='real_images')
self.G = self.network(self.inputs)
self.sess.run(self.G, feed_dict={self.inputs: inputs})
```

Say you're like me and you tend to run networks for a number of epochs and then drop into 
a debugger sesssion (via `pdb.set_trace()`). Now, these epochs could constitute hours or 
even days of computing time, and you realize when you try to apply your newly trained
filters onto some different data and you get the following 

```
InvalidArgumentError (see above for traceback): Conv2DSlowBackpropInput: input and out_backprop must have the same batch size
     [[Node: conv2d_transpose_3 = Conv2DBackpropInput[T=DT_FLOAT, data_format="NHWC", padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/gpu:0"](conv2d_transpose_3/output_shape, w0/read, _recv_lg_real_images_0/_69)]]
     [[Node: Tanh_1/_71 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_4087_Tanh_1", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]
```

Oops! If you restart the application then you'll lose your filters (assuming you haven't 
saved them already.) What you can do is redefine your `batch_size`. Rerunning 
`self.sess.run(self.G, feed_dict={self.inputs: inputs})` will yield the same result, 
however. So what's to be done? Simply reload the method onto the tensorflow graph
via `self.G = self.network(self.inputs)` and then you'll be good to go.
