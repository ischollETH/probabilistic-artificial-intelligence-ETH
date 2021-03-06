# probabilistic-artificial-intelligence-ETH
Python projects for the course 'Probabilistic Artificial Intelligence' (Prof. Krause) 2021 @ ETH Zurich. All of the projects have outperformed the requested score baseline by far and therefore passed all the tests, resulting in an overall project grade of 6/6.

Credit for the project setup and skeleton codes goes to Prof. Krause and his teaching assistants ([course website](https://las.inf.ethz.ch/teaching/pai-f21)).

## Task 0: Bayesian Inference
Simple introductory exercise computing posterior probabilities based on different scaling hypothesis spaces. 

## Task 1: Gaussian Process Regression
Task implementing a gaussian process regression in order to infer air pollution in different places based on spatial data. The [KFold method by sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) has been used in order to cope with the very large amount of data. This allowed to approximate an otherwise hard to compute model posterior. The following visualizes some predictions of the fitted model: 
<p align="center">
  <img src=https://github.com/ischollETH/probabilistic-artificial-intelligence-ETH/blob/main/images/task1_extended_evaluation.png width="1000" title="Visual representation of GP means and standard deviations">
</p>

## Task 2: Bayesian Neural Nets
Task implementing a Bayesian Neural Network, employing Univariate and Multivariate Gaussians and using negative log-likelihood as well as the Kullback-Leibler divergence for loss calculations. The usecase was training and testing on the [rotated MNIST dataset](https://github.com/ChaitanyaBaweja/RotNIST) as well as the [fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). Such a network not only achieves good prediction accuracies, but at the same time will also compute a certainty for its predictions, which can be very useful for many usecases in order to know how much the prediction results can be relied upon. First, some excerpts of the performance on the rotated MNIST dataset can be observed, where either the network was not confident in its prediction (and therefore sometimes wrong, too) or very confident (and correct); respectively some ambiguous data samples where the ground truth is not known for sure:
<p align="center">
  <img src=https://github.com/ischollETH/probabilistic-artificial-intelligence-ETH/blob/main/images/task2_mnist_least_confident.png width="700" title="Least confident predictions on some MNIST dataset samples">
</p>
<p align="center">
  <img src=https://github.com/ischollETH/probabilistic-artificial-intelligence-ETH/blob/main/images/task2_mnist_most_confident.png width="700" title="Most confident predictions on some MNIST dataset samples">
</p>
<p align="center">
  <img src=https://github.com/ischollETH/probabilistic-artificial-intelligence-ETH/blob/main/images/task2_ambiguous_rotated_mnist.png width="700" title="Some ambiguous MNIST samples and their predictions">
</p>

Further, some similar excerpts of the results on the fashion MNIST dataset can also be observed:

<p align="center">
  <img src=https://github.com/ischollETH/probabilistic-artificial-intelligence-ETH/blob/main/images/task2_fashionmnist_least_confident.png width="700" title="Least confident predictions on some Fashion MNIST dataset samples">
</p>
<p align="center">
  <img src=https://github.com/ischollETH/probabilistic-artificial-intelligence-ETH/blob/main/images/task2_fashionmnist_most_confident.png width="700" title="Most confident predictions on some Fashion MNIST dataset samples">
</p>


## Task 3: Bayesian Optimization
Task implementing a Bayesian Optimization (minimization) algorithm using Gaussian Processes for the constraint and objective function. All three common acquisition functions following the paper by [Snoek et al. (2012)](https://arxiv.org/pdf/1206.2944.pdf) were tested: Probability of Improvement (PI), Expected Improvement (EI) and Lower Confidence Bound (LCB). This allows to tune hyperparameters all the while constraints are active. The best results were observed when using EI, and in the following visualizations of the posteriors of both the constraint as well as objective function are shown:

<p align="center">
  <img src=https://github.com/ischollETH/probabilistic-artificial-intelligence-ETH/blob/main/images/task3_extended_evaluation.png width="1000" title="Visual representation of GP posterior for the objective and constraint function">
</p>


## Task 4: Reinforcement Learning (RL)
Task implementing a Reinforcement Learning (RL) algorithm that, by practicing on a simulator, learns a control policy for a lunar lander. The approaches used and studied were (1) using only policy gradients, (2) using rewards-to-go on top and (3) generalized advantage estimation as a baseline. The latter two are used to reduce the variance of the policy gradients by ignoring noisy or pooling noisy rewards from the past which were not a consequence of the action under consideration. The following shows some resulting runs of the simulation after training, where the lunar lander is supposed to land between the yellow flags without crashing:

<p align="center">
  <img src=https://github.com/ischollETH/probabilistic-artificial-intelligence-ETH/blob/main/images/task4_policy.gif width="80%" title="Several sequences of the spaceship landing trained through reinforcement learning">
</p>


