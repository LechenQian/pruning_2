

# How light can a neural net be


I would like to deep dive into the paper from Han lab - "Learning both Weights and Connections for Efficient Neural Networks". This study proposes a three-step method for training lightweight neural networks that both reduce model size and latency and can be deployed on-device while maintaining high accuracy. I will showcase the effect of pruning based on a small model and a toy dataset.

## Contents:
- Background of this work
- Understand the basic concept of **pruning**
- Three-step Training Pipeline for Training Efficient Neural Networks

- My implementation of **fine-grained pruning**

- Get a basic understanding of performance improvement (such as speedup) from pruning

## Background of this work
**Neural networks have become ubiquitous in many applications. While large neural networks are very powerful, their size consumes considerable storage, memory bandwidth, and computational resources.**

Neural networks are challenging to implement on mobile devices due of their intensive computational and memory requirements. Additionally, the architecture was predesigned for the network prior to training, which prevents training from improving the architecture. In order to overcome these restrictions, this study outline a pruning technique for learning only the crucial connections, which allows neural networks to store and compute an order of magnitude less data while maintaining their accuracy. This training process learns the network connectivity in addition to the weights very similar to human brain: during brain matures, some neurons will be lost through apoptosis, which helps to shape the brain and create functional neural networks.

## Three-step Training Pipeline for Training Efficient Neural Networks
Here are three major steps to make the neural network more light-weight:
1. first train the network to identify the most crucial connections. 
1. remove the unnecessary connections. 

    You prune the connections that have low weights or in other words at least the network considers those weights to be not important during the training phase and once the pruning of those weights is done you'll have a less dense and relatively smaller network 
1. retrained the network to adjust the weights of the remaining connections.

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/pruning_diagram.jpg" width="100%" />

### results from paper
The researchers demonstrate through experimentation that the AlexNet and VGG16 models on the ImageNet dataset can be reduced to 9x and 13x of their original sizes while still being able to provide accuracy values that are almost identical.


<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/pruning_result_paper.jpg" width="100%" />




## My implementation of **fine-grained pruning**
### - Evaluate the Accuracy and Model Size of a Dense Model

To first evaluate the accuracy and model size of a relatively small neural network, I train a 5-hidden-layer CNN toy model on the MNIST dataset.
    
    class Net(nn.Module):
    # Constructor
        def __init__(self, num_classes=3):
            super(Net, self).__init__()
            
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
            self.conv3 = nn.Conv2d(32,64, kernel_size=5)
            self.fc1 = nn.Linear(3*3*64, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            #x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(F.max_pool2d(self.conv3(x),2))
            x = F.dropout(x, p=0.5, training=self.training)
            x = x.view(-1,3*3*64 )
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

After just five epochs of training, this toy dense CNN model has accuracy=99.29% on the MNIST test set with
size=0.87 MiB. 

#### 
The number of parameters across different layers varies substantially as shown below:
<p align="center">
    <img src="https://github.com/LechenQian/pruning_2/blob/main/figures/number_parameters_prepruned.png" width="50%" />
</p>

## Number of parameters in each layer
the number of each layer's parameters also affects the decision on sparsity selection. Layers with more #parameters require larger sparsities.

#### The distribution of weight values in the dense model

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/histogram_prepruned_weights.png" width="100%" />

The distribution of weights is centered on zero with tails dropping off quickly. Weights that are close to 0 are the ones that are lease important, which helps alleviate the impact of pruining on model accuracy.

Though I used a fairly small CNN network as an toy example here, you will still be amazed by how much redundency it has from the following pruning steps. Of course, the goal of pruning is to reduce the model size while maintaining the accuracy.

### Fine-grained pruning
Fine-grained pruning removes the synapses with lowest importance. The weight tensor $W$ will become sparse after fine-grained pruning, which can be described with **sparsity**:

> $\mathrm{sparsity} := \#\mathrm{Zeros} / \#W = 1 - \#\mathrm{Nonzeros} / \#W$

where $\#W$ is the number of elements in $W$.

#### **Maginitude-based pruning**
For fine-grained pruning, a widely-used importance is the magnitude of weight value, *i.e.*,

$Importance=|W|$

This is known as **Magnitude-based Pruning**


#### **Sensitivity scan**

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/sensitivity_curves.png" width="100%" />

The relationship between pruning sparsity and model accuracy is inverse. When sparsity becomes higher, the model accuracy decreases.

not all of the layers showing the same sensitivity. From the plot, we can see that the first convolution layer (backbone.conv0) is the most sensitive tot he pruning sparsity.

#### **Decide the sparsity for each layer**
    sparsity_dict = {
    'conv1.weight': 0.65,
    'conv2.weight': 0.95,
    'conv3.weight': 0.98,
    'fc1.weight': 0.98,
    'fc2.weight': 0.9
    }

I picked a set of aggresive sparcities and now the sparse model has 0.02 MiB in size, which is 2.71% of original dense model size and the accuracy is  only about 8.85% after pruning. This is not surprising that the accuracy dignificantly dropped after an aggresive pruning, which also necessitate the fine tuning step.

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/histogram_pruned_weights.png.png" width="100%" />

### Retraining the network
    Finetuning Fine-grained Pruned Sparse Model
    Epoch 1 Sparse Accuracy 96.76% / Best Sparse Accuracy: 96.76%
    Epoch 2 Sparse Accuracy 97.03% / Best Sparse Accuracy: 97.03%
    Epoch 3 Sparse Accuracy 97.43% / Best Sparse Accuracy: 97.43%
    Epoch 4 Sparse Accuracy 97.42% / Best Sparse Accuracy: 97.43%
    Epoch 5 Sparse Accuracy 97.39% / Best Sparse Accuracy: 97.43%
sparse model has accuracy=97.76% and the model size now is 0.02 MiB, which is 36.88X smaller than the 0.87 MiB original already small CNN model. This shows that even a small model has a great redundency and a big room for pruning. 



