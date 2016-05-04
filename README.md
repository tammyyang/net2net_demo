# net2net_demo
This repository includes the demo scripts to use Net2Net method to make net deeper and wider
on Keras framework. The Net2Net class was implemented by [Kyunghyun Paeng](https://github.com/paengs/Net2Net) in python
which can take any weight matrix as input and return the optimized weights. 
The reference of the demo net was from the [mnist_cnn.py](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) in the Keras examples.

## Usage
```
THEANO_FLAGS=exception_verbosity=high,device=gpu,floatX=float32 python mnist_net2net.py
```

The script 
