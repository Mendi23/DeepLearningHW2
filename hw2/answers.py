r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.01
    reg = 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.012
    lr_momentum = 0.005
    lr_rmsprop = 0.0005
    reg = 0
    
    
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. Yes, the graphs match what we expected to see. the lower the dropout, the bigger the overfitting is. in the first graph we can observe that there's quite extensive overfitting when the dropout is zero (the model increase its accuracy on the training set - 90 precent - but not on the test set - 20 percent). the more we increase the dropout portion, we get less overfitting, until we get real underfitting with the highest dropout.

2. we can see from the graphs that the low-dropout model is as we say overfiiting, and the high-dropout model is underfitting since the acc on the train and the test sets is low. we can infer from that that he model was to general and diddn't succeded learning the data well.

"""

part2_q2 = r"""
**Your answer:**

Yes, it is possible for the accuracy to increase while the loss is also increasing for a few ephoc. since the accuracy is zero-one loss, which means that it only counts the number of succeses while the loss calculates a function of the distances from the correct answer. so if in the learnign process the model decreased the number of errors (in 0-1 terms) but "pushed" all the data points farther from the imaginary hyperplane, so that the remaining error's distance increases, the described phenomena will occur. 

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
First, We'll explain how we chose the parameters for our model.


## Hyperparameters selection


### CNN

- hidden dimentions: using plural (i.e. 3 or 4) linear layers with high output-dimensional seemed to have positive affect, but we got good result even when used one fc layer with 50 outputs. We notiched a trend where hidden-dims of the shape $medium,small,large$ outpreformed other combinations of fc layers, so we defined 3 fc layers with dims: `100, 50, 512`.

- padding: was too difficult to parse changes per-layer. we examined fixed values for all layers and eventually came back to `0` padding.

### Trainer

- batch size: The training was ineffective when using large batches, beacuse of memory issues and was too time consuming (as expected). we settled on mini-batches of `50` examples each.

- dataset size: The tuning and optimizing was done with 12K examples. for the final model for the experiments we used `30K` examples.

- number of iterations: in all cases we noticed that the there was virtually no changes after ~150 epochs, so we ran the trainer `200` with early stopping after `10` iterations withous change.

### Optimizer

- optimization algorithm: The `Momentum-SGD` preformed better on the tasks we experimented on. we didn't bother too much with the momentum factor and used `0.9` as is the common practice.

- regularization factor: as we increased its value, the convergence was steeper, but the accuracy didn't imporve. when we tried decreasing the value, the train acc still converged to avout 100% but without meaningfull change in the test acc. `0.8` gave us a pretty fast convergence with relatively small overfitting (even though still quite pronounced, as can be seen from the difference between train and test scores).

- learning rate: different values (between 0.01 and 0.0001) didn’t seem to have much of an effect on the result, other than the graph shape (convergence rate) of course. we settled on the middle: `0.001`.


### Important explanation:
Because of time constraints, we optimized for the `L=2` with randomized `k` from the different values in the experiments, **while the same parameters are used for all configuration**. The final parameters were chosen very much using a combination of "greedy search" (because we changed one or two parameters at a time), "randomized search" (because the rest of the parameters were chosen at random) and the eye test (we estimated the optimal values by looking at the results and graphs). iteratively, we changed manualy a couple of paramteres, tested them on a couple of random architucture (with 2 layers per block) and settled on what seems to yeald the better results. hardly "optimal" but this method did helped us improved the model's preformance in limited time and computing resources.


## Experiment results

We can see (maybe counterintuitively) that there's reverse correlation between depth and preformences. meaning, the more layers we add (the bigger `L` is), the model does worse.

The graphs practically "scream" overfiting. after the first couple iterations (around 25), the training-loss is optimizing for the cost of increasing the test-loss (which just shoot up), even though test-accuracy remain more-or-less the same. that's because the model have many features (weights) to train, resulting in a model tailor-made for the training-set (getting to 100 accuracy very fast), on the expense of generalizing. deeper model has more trainable-features, which increase the chance of overfiting (as can be seen clearly).

The 16-layer model was untrainable (as we can see, there's no change in the blue line on either plot). the cause is most likely kind of "vanishing gradient". multiplying many gradients (very small values) in the backpropagation process caused the gradient to zero. possible solutions are batch normalization and residual network.

Increasing the number of filters from 32 to 64 gave a small boost to preformances. the overfiting problem still seems as prevalent, so the small bump can be attributed to the (somewhat accidental) capturing of general features in the trainng set. 

We can also we in the graph the effect of "early stopping" when the test-loss consistenly increases (in the future, it's worse exploring more dynamic early stoping, based on averages and such, in order to stop the trainig based on the overall trend of the loss function and avoid reseting the epoc counter when there's a small drop i otherwise Increasing function).

"""

part3_q2 = r"""
**Your answer:**

The impact of the filter count seem to be much more direct and pronounce then the depth. in all cases, increasing the number of filters per layer resulted in better accuracy.

This results hold for all depths. looking at the graphs can further validate the conclusions from the previous experiment. more layers resulted in less accuracy, and all evidence (such as graph shape) are pointing to overfiting.

"""

part3_q3 = r"""
**Your answer:**

This experiment encompass all the ideas depicted in the previous 2 sections. adding filters seems to have positive effect (this configuration yealded the best results), while adding layers let to the opposite outcome. The different between the train and test scores can be attributed to quite massive overfitting.

The deepest NN (12 and 16 layers) suffered from vanishing gradients and couldn't be trained properly.

"""


part3_q4 = r"""
**Your answer:**
## CNN Architecture

We chose to base our implementation on the ResNet18 architecture (as presented in the tuturial) because the number of filters in each block of this architecture matches the requierments of the second experiment.
we ignored the input layer (in order not to change the depth specified in the experiment parameters) and defined two types of blcoks:

- ResNetConvBlok: **Conv2d** layer with 3x3 kernels, padding and stride of 1, followed by **BatchNorm2d**. the *forward* method of the block also implement a "shortcut" (circonvene the layer altogether), as shown in the tuturial.

- ResNetPullBlock: **MaxPool2d** with 2X2 kernels, padding of 1 and stride of 2, followed by a **Dropout** layer.

The parameters for each block are based on similar blocks in known archituctures found online. extensive optimization efforts also validate our choices.

The feature_extractor is a sequence of `L` ResNetConvBlok blocks (with the same number of filters in the layers of each block) followd by ResNetPullBlock (pool_every = `L`), followd by **AdaptiveAvgPool2d** layer with 1X1 kernels, used to normalize the output dimensions.

The classifier is a single fc layer which transform the output diimension to the number of classes (we didn't add any hidden dimentions).

We've also implemented *dynamic dropout*. meaning, the droput factor is 0.2 in the first dropout layer, and is increasing every layer up to 0.5 (learned the idea online and found that is offers slight improvment in preformence). 

From the begining it was clear that the bigget issue is overfitting (the acc of the trainng data always converged to 100), and that was the guiding principle in desining the NN, e.g. avoidence from adding hidden layers, allowing "shortcuts", using dropout, pooling etc. this logic also led us to select large regularization factor (6.0).

## Results

We can see that in all cases, the accuracy is stabilized after ~100 iterations. from this point, the optimizer tries more "exploration" (as can be seen in the graph of the training-data's loss function), which led to a small overall reduction in the loss function, but doesn't affect the accuracy score.

All models preformed much better then their counterparts in experiment 1, especially the multylayer models which approached 90 percent accuracy. this is probably mostly due to our efforts at limiting overfitting and the depth of the NN.

There is virtualy no notable difference between the multylayer models, regardless of their actual depth. this can be attributed to the "shortcut" feature of ResNet. we can assume that the optimizer sees no utility in adding more layers beyond those in the model where `L=2`, so it just chooses to "skip" the extra layers in the deeper models. even when `L=1` (and the model has just 4 conv layers), the test score is well above 80 percent, which strengths our assumption that depth matter only up to a certain threshold (5-6 layers in this case). 

The use of ResNet (and the ability to "skip" over blocks) also neutralize the problem of "vanishing gradients" from experiment 1, which allow us to train a deeper model. 

The shape of the loss function of the test-set indicate that it could be benefial to try other types of loss functions, as well as adjusting regularization, in order capture better approximation of "accuracy". 


"""
# ==============
