
## A C implemention for convolutional network

This is a project for implementing a convolutional network by C, the convolutional network is
constructed as following:

![net_structure](./img/net_structure.png) 

The key kernel of convolutional network is the forward and back propagation of convolutional layer. 
We first give the formula of these two processes.

![conv_structure](./img/conv_structure.png)

Let W be the input size, F be the filter size, S be the strider, and P be the output size, 
and we assume no padding here. Then W, F, S, P satisfy:

![formula_1](./img/formula_1.png)

### Forward propagation
The forward propagation is the convolution of the input and the filter as following:

![formula_2](./img/formula_2.png)

### Back propagation
The back propagation is the transfer of gradient by chain of rule, we have:

![formula_3](./img/formula_3.png)

### Loss function
The loss function L is the softmax\_cross\_entropy of logits, we also give the formula of its forward and back propagation.

#### softmax_forward:

![softmax_forward](./img/softmax_forward.png)

#### softmax_backward:

The softmax_backward depends labels, we have

![softmax_backward_1](./img/softmax_backward_1.png)

if j doesn't equal to lb,

![softmax_backward_2](./img/softmax_backward_2.png)

if j equals to lb,

![softmax_backward_3](./img/softmax_backward_3.png)

#### cross\_entrop\_forward:

![cross\_entrop\_forward](./img/cross_entropy_forward.png)
