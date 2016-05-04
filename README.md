# net2net_demo
This repository includes the demo scripts to use Net2Net method to make net deeper and wider
on Keras framework. The Net2Net class was implemented by [Kyunghyun Paeng](https://github.com/paengs/Net2Net) in python
which can take any weight matrix as input and return the optimized weights. I found this implementation is very useful because it is independent of the deep learning framework. 
The reference of the demo net was from the [mnist_cnn.py](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) in the Keras examples.

## Usage
```
THEANO_FLAGS=exception_verbosity=high,device=gpu,floatX=float32 python mnist_net2net.py
```

The script inserts a fully connected layer after the 7th layer of the original net. If you are interested in replacing it, find `new_layer = (7, Dense(128))` in `mnist_net2net.py` and replace it with ($INDEX, $LAYER) you want. 

## Result

![](dist_1d.png)
