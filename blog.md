

# How light can a neural net be

Selina Qian

Dec 12, 2022

## Contents
- [Background of this work](https://github.com/LechenQian/pruning_2/blob/main/blog.md#background-of-this-work)
- [Three-step Training Pipeline for Training Efficient Neural Networks](https://github.com/LechenQian/pruning_2/edit/main/blog.md#three-step-training-pipeline-for-training-efficient-neural-networks)
- [My implementation of **fine-grained pruning**](https://github.com/LechenQian/pruning_2/edit/main/blog.md#my-implementation-of-fine-grained-pruning)


## Background of this work
Many applications now use neural networks. Huge neural networks are extremely strong and they have been applied successfully to speech recognition, image analysis and adaptive control. However, the considerable storage, memory bandwidth, and computing resources required by neural networks make it challenging to implement on mobile devices. For instance, running a 1 billion connection neural network at 20Hz would require 12.8W for DRAM access - well beyond the power envelope of a typical mobile device. 

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/size_current.jpg" width="100%" />

Additionally, the architecture was predesigned for the network prior to training, which prevents training from optimizing the architecture. In order to overcome these restrictions, this study[1] outline a pruning technique for learning only the crucial connections, which allows neural networks to store and compute an order of magnitude less data while maintaining their accuracy. Artificial neural networks, which loosely model the neurons in a biological brain. Interestingly, this training process learns the network connectivity and weights very much similar to human brain[2]: during brain matures, some neurons will be lost through apoptosis, which helps to shape the brain and create functional neural networks.


I would like to deep dive into the paper from Han, Song, et al. - "Learning both Weights and Connections for Efficient Neural Networks". This study proposes a three-step method for training lightweight neural networks that both reduce model size and latency while maintaining high accuracy. I will showcase the effect of pruning using a small CNN model on MNIST dataset.

## Three-step Training Pipeline for Training Efficient Neural Networks

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/pruning_diagram.jpg" width="100%" />


Here are three major steps to make the neural network more light-weight:
1. First train the network to identify the most crucial connections. 
    * Difference is that instead of training the network to get the final weights, they are learning which connections are important by setting certain thresholds! The criteria used inthis study for determining the importance of weights is the magnitude of weight value, *i.e.*, $Importance=|W|$

1. Remove the unnecessary connections. 

    * You prune the connections that have low weights or in other words at least the network considers those weights to be not important during the training phase. The approch is that you first define the sparsities of each layer or the whole network. Then we can find the corresponding weight threshold for pruning. Fine-grained pruning removes the synapses with low importance below threshold. Once the pruning of those weights is done you'll have a less dense and relatively smaller network. 
1. Fine tune the remaining weights on the smaller network and repeat the step 2 Until reach the certain threshold of the performance
    * This step is necessary and critical because without retraining, the accuracy of the network is significantly negatively impacted as I will demonstrate in my implementation.
    * **Parameter Co-adaptation**: Importantly, as discussed in the paper, while doing retraining, it's better to keep the surviving trained parameters instead of reinitializing the parameter. The rationale is that random initialization will disrupt the already-good parameter solution foudn by the initial training process. In the case of CNNs, they contain fragile co-adapted weight features and it will better to keep them while retraining.
    * Iterative Pruning gives better result on finding the proper connections. What we can do is that run more than one iterations involves pruning and retraining. In this way, we might be able to find the potentially minimal number of connections after several rounds of iterations. And this method has proven to be reducing the model size more without losing accuracy then one-step aggresive pruning.




### Results from paper
The researchers demonstrate through experimentation that the AlexNet and VGG16 models on the ImageNet dataset can be reduced to 9x and 13x of their original sizes while still being able to provide accuracy values that are almost identical. 


<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/pruning_result_paper.jpg" width="100%" />

As can be seen in the picture below, the CONV and FC layers respond to pruning in distinct ways. We can observe how accuracy decreases as parameters are trimmed layer by layer. Compared to completely connected layers, the CONV layers (on the left) are more susceptible to pruning (on the right). The first convolutional layer is the one that is most susceptible to pruning among the CONV layers. In general, fully linked layers have more connections and plenty of room for pruning to cut down on redundancy without hurting the accuracy.

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/sensitivty_paper.jpg" width="100%" />

After pruning, AlexNet and VGGNet's storage needs are so minimal that all weights may be stored on the chip rather than off-chip DRAM.


<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/comparison_models.jpg" width="100%" />



## My implementation of **fine-grained pruning**
### - Evaluate the Accuracy and Model Size of a Dense Model

To first evaluate the accuracy and model size of a relatively small neural network, I trained a 5-hidden-layer CNN toy model on the MNIST dataset.
    
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

After just five epochs of training, this dense CNN model has accuracy=99.29% on the MNIST test set with
size=0.87 MiB. 

#### **Number of parameters in each layer**

Given this trained model, we could check the number of parameters across different layers. The numbers vary substantially as shown below:
<p align="center">
    <img src="https://github.com/LechenQian/pruning_2/blob/main/figures/number_parameters_prepruned.png" width="50%" />
</p>

The number of each layer's parameters also affects the decision on sparsity selection as we can roughly infer how much redundancy lies in each layer. Layers with more parameters will require a bigger pruning strength.

#### **The distribution of weight values in the dense model**

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/histogram_prepruned_weights.png" width="100%" />

The distribution of weights is centered on zero with tails dropping off quickly. Weights that are close to 0 are the ones that are lease important, which helps alleviate the impact of pruining on model accuracy.

Though I used a fairly small CNN network as an toy example here, you will still be amazed by how much redundency it has from the following pruning steps. Of course, the goal of pruning is to reduce the model size while maintaining the accuracy.

### Fine-grained pruning
I perform fine-grained pruning based on the sparsities I pre-defined for each layer.
 **sparsity**: 
 $$\mathrm{sparsity} := number\ of\ \mathrm{Zeros} / number\ of\ W = 1 - number\ of\ \mathrm{Nonzeros} / number\ of\ W$$

where number of $W$ is the number of elements in $W$.


#### **Sensitivity scan**
I runned a set of sparsities for pruning on a layer-by-layer basis and check the relationship between accuracy and layer sparsity.

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/sensitivity_curves.png" width="100%" />

There is clearly an inverse relationship between pruning sparsity and model accuracy: when sparsity becomes higher, the model accuracy decreases. As expected from the paper, not all of the layers showing the same sensitivity in my implementation. From the plot, we can see that the first convolution layer is also the most sensitive to the pruning sparsity. And FC layers with much more number of weights are more immune to pruning.

#### **Decide the sparsity for each layer**
    sparsity_dict = {
    'conv1.weight': 0.65,
    'conv2.weight': 0.95,
    'conv3.weight': 0.98,
    'fc1.weight': 0.98,
    'fc2.weight': 0.9
    }

Since I only plan to do one iteration of pruning as demonstration purpose, I picked a set of aggresive sparcities. Now the sparse model has 0.02 MiB in size, which is 2.71% of original dense model size. Unfortunately the accuracy is only about 8.85% after pruning. This is not surprising that the accuracy dignificantly dropped after an aggresive pruning, which also necessitate the fine tuning step.

<img src="https://github.com/LechenQian/pruning_2/blob/main/figures/histogram_pruned_weights.png" width="100%" />

### Retraining the network
    Finetuning Fine-grained Pruned Sparse Model
    Epoch 1 Sparse Accuracy 96.76% / Best Sparse Accuracy: 96.76%
    Epoch 2 Sparse Accuracy 97.03% / Best Sparse Accuracy: 97.03%
    Epoch 3 Sparse Accuracy 97.43% / Best Sparse Accuracy: 97.43%
    Epoch 4 Sparse Accuracy 97.42% / Best Sparse Accuracy: 97.43%
    Epoch 5 Sparse Accuracy 97.39% / Best Sparse Accuracy: 97.43%
A sparse model has accuracy=97.76% and the model size now is 0.02 MiB, which is 36.88X smaller than the 0.87 MiB original already small CNN model. This shows that even a small model has a great redundency and a big room for pruning. 

## Reference
[1] Han, Song, et al. "Learning both weights and connections for efficient neural network." Advances in neural information processing systems 28 (2015).

[2] Rauschecker, J. P. "Neuronal mechanisms of developmental plasticity in the cat's visual system." Human neurobiology 3.2 (1984): 109-114.

