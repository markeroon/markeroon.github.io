---
layout: post
title: Issue using decode_jpeg
---

Here's a weird issue that I came across today when using queues with Tensorflow. If you get the error

```
W tensorflow/core/framework/op_kernel.cc:975] Invalid argument: Shape mismatch in tuple component 0. Expected [224,224,3], got [224,224,1]
```

when using `decode_jpeg`, you likely need to specifiy the number of channels you're using, i.e. `image = tf.image.decode_jpeg(image_file, channels=3)`.

This seems completely insane to me, but here we are.
