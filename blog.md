you will practice pruning a classical neural network model to reduce both model size and latency. The goals of this assignment are as follows:

- Understand the basic concept of **pruning**
- Implement and apply **fine-grained pruning**
- Implement and apply **channel pruning**
- Get a basic understanding of performance improvement (such as speedup) from pruning
- Understand the differences and tradeoffs between these pruning approaches

# Let's First Evaluate the Accuracy and Model Size of Dense Model
Neural networks have become ubiquitous in many applications. Here we have loaded a pretrained VGG model for classifying images in CIFAR10 dataset.

Let's first evaluate the accuracy and model size of this model. 

While large neural networks are very powerful, their size consumes considerable storage, memory bandwidth, and computational resources.
As we can see from the results above, a model for the task as simple as classifying $32\times32$ images into 10 classes can be as large as 35 MiB.
For embedded mobile applications, these resource demands become prohibitive.

Therefore, neural network pruning is exploited to facilitates storage and transmission of mobile applications incorporating DNNs.

The goal of pruning is to reduce the model size while maintaining the accuracy.

# Let's see the distribution of weight values
Before we jump into pruning, let's see the distribution of weight values in the dense model.

**[image of weights distribution]**

description of image:The distribution of weights is centered on zero with tails dropping off quickly. Weights that are close to 0 are the ones that are lease important, which helps alleviate the impact of pruining on model accuracy.

# Fine-grained pruning
Fine-grained pruning removes the synapses with lowest importance. The weight tensor $W$ will become sparse after fine-grained pruning, which can be described with **sparsity**:

> $\mathrm{sparsity} := \#\mathrm{Zeros} / \#W = 1 - \#\mathrm{Nonzeros} / \#W$

where $\#W$ is the number of elements in $W$.

## Maginitude-based pruning
For fine-grained pruning, a widely-used importance is the magnitude of weight value, *i.e.*,

$Importance=|W|$

This is known as **Magnitude-based Pruning**


## Sensitivity scan

**[sensitivity curve]**
The relationship between pruning sparsity and model accuracy is inverse. When sparsity becomes higher, the model accuracy decreases.

not all of the layers showing the same sensitivity. From the plot, we can see that the first convolution layer (backbone.conv0) is the most sensitive tot he pruning sparsity.

## Number of parameters in each layer
In addition to accuracy, the number of each layer's parameters also affects the decision on sparsity selection. Layers with more #parameters require larger sparsities.

**[number of parameters]**
