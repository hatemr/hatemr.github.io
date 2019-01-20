---
Title: Overview of Machine Learning
Date: 2018-08-12 22:10
Author: Robert Hatem
Lang: en
Tags:
Summary: Tutorial on common statistical modeling techniques
Image: images/CRA.img
mathjax: true
---

Machine learning has become popular recently as  more data becomes available for analysis.  This set of techniques are not new, but they are more useful now given more data.

These techniques go by different names in different academic and professional communities.  Computer scientists call them *machine learning*, and the "data science" community has done the same.  The statistics community calls them statistics or statistical learning.  In banking and finance, they're often called *modeling* [^1].  Some prefer a compromise: **statistical modeling**.

Using the popular name, machine learning can be broken down into sets: supervised and unsupervised learning.

## Supervised Learning
Supervised learning is when we predict a target variable $y$ using predictor variables $x_i$ .  We have the *correct* answer in our training data: the $y$ values.  The training data is "labeled" with the correct answer.  We train our model on the training data, then make predictions on new examples using $x$ to predict $y$, which is unknown.

Supervised learning tries to find the function $f(x)$, where $f$ is a mapping from $x$ to $y$: $y=f(x)+\epsilon$.

Supervised learning can be split into two separate tasks: regression and classification.

1. **Regression**: target variable $y$ is continuous (e.g. annual salary, home values in USD).  In other words, it maps a real-valued function that maps an n-dimensional space $\mathbb{R}^n$ onto the real line $\mathbb{R}$.  $f: \mathbb{R}^n \rightarrow \mathbb{R}$.  Given pairs $(X_i, Y_i)$. 
2. **Classification**: target variable $y$ is categorical. (e.g. default/non-default, low/medium/high income).  $f: \mathbb{R}^n \rightarrow \{1,...,k\}$.  Also Given pairs $(X_i, Y_i)$.

**Common methods**:

* Linear regression
* Logistic Regression
* Decision Trees (random forests, gradient boosting, ensemble methods)
* Support Vector Machines (SVMs)
* K-nearest neighbors
* Feedforward, convolutional, and recurrent neural networks


## Unsupervised Learning

In unsupervised learning we don't have the correct answers, we have only the $x$ variables.  Instead of prediction, we seek to find structure in the data.  One example is clustering; splitting the data into a few groups.  Since there is no "correct" answer, there is no straightforward way to assess the accuracy of unsupervised models; this must rely on domain knowledge.

Two common tasks in unsupervised learning are clustering and dimensionality reduction:

1. **Clustering**: Splitting the data into a few groups (e.g. splitting customer into groups based on demographics and purchasing behavior to serve them more relevant ads). We need to learn the function $f: \mathbb{R}^n \rightarrow {1,...,k}$ given only $(X_i)$, and not given the $k$ categories, unlike in classification.
2. **Dimensionality Reduction**: Represting the data in a compact, reduced form, which can save file space.  Learn the function $f: \mathbb{R}^n \rightarrow \mathbb{R}^k$ which maps the n-dimensional data onto a lower dimensional space, where $k$ is lower than $n$, and we are given only $(X_i)$.

**Common methods**:

* K-means clustering
* Hierarchical clustering
* Principal Componenets Analysis
* Gaussian mixutre modeling
* Hidden Markov models
* Autoenconders (neural networks)
* Generative Adversarial Networks (GANS; neural networks)


## Reinforcement Learning

In reinforcement learning, there is a 'correct' answer given, but only after a number of decisions were made.  In this sense it's halfway between supervised and unsupervised learning; it gives feedback, but only partially.

The goals is to learn a policy function $f: \mathbb{R}^n \rightarrow \mathbb{R}^k$, which takes in a state $X_i$, chooses an action $a_i$, then receives a reward $r_i$, and arrives at the next state $X_{i+1}$.  As such we are given tuples $(X_i, a_i, r_i, X_{i+1})$.

## Bias-variance Tradeoff
Bias and variance are two competing properties of a statistical model - they must be traded-off to achieve optimal performance.

* **Variance**: refers to the amount by which our estimated model $f$ would change if we estimated it using a different data set.
* **Bias**: refers to the error that is introduced by approximating a real-life problem, which may be very complicated, using a much simpler function.

We trade off between bias and variance to optimize performance on new, unseen data.  We aim for a model generalizes to new data.

In general, highly flexible models have low bias but high variance - they will fit any given data set well, but will give different models when estimated on different data sets.  Conversely, models with low flexibility have high bias and low variance - they will not fit any given data set well, but the estimated model won't vary much between different data sets.  For example, linear regression is relatively inflexible while neural networks are highly flexible - they can fit a much wider range of functions.

## Training, Validation, and Test Sets
When building a statistical model, the data must be split into 3, sometimes 2, data sets:

* **Training set**: Used to estimate the model parameters (e.g. coefficients in linear and logistic regression).
* **Validation set**: Used to estiamte *hyperparamters* (e.g. regularization parameter in Lasso and Ridge regression, number of layers, and rest of network architecture, in deep neural networks.)
* **Test set**: Used to assess the model's accuracy on new, unseen data.

## Overfitting
Overfitting occurs when a model fits the training set well, but fails to predict well on the test set.  This indicates the model is too flexible; it adjusted to fit the training data *too* well, and failed on new data.  The solution is to reduce the model's flexibility; this is called regularization.

For example, linear regression can be regularized using the L1 norm (Lasso) and L2 norm (Ridge) on the cost function, limiting the possible values the coefficients can take on.  This will raise the model's bias, but lower its variance, hopefully reducing the test error.  The amount of regularization is adjusted to minimize the test error.

Neural networks are highly flexible models which must be carefully checked, and regularized, to avoid overfitting.  Though their flexibility makes them difficult to understand, like a black-box, when appropriately regularized and applied to the correct problems neural networks offer highly accurate predictions.

## Is Linear Regression part of Machine Learning?
Yes!  Linear regression is a technique in machine learning.  The term *machine learning* is basically a buzzword for statistics.  Machine learning is basically the computer science community's word for statistics.  It's just a buzzword.  For someone who is unconvinced, notice that linear regression is just a single-layer neural network with the identity activation function; if deep neural networks count as machine learning then so does linear regression.

### Buzzwords
There are a lot of buzzwords flying around; it can be hard to know what it all means.  Here are some common buzzwords and what they actually mean. 
* Artificial Intelligence - Machine achieving human-level performance at some specific task (face recognition, credit approval, recognizing speech)

* Machine Learning - Subset of AI that teaches a computer to perform a task from experience (i.e. data), rather than having it pre-programmed beforehand by a human.

* Data Mining - Uses ML to find patterns in data in a quest for actionable ideas.

* Big Data - Data mining on large sets of structured and unstructured (text, speech) data

* Data Science - Science of performing data mining on Big Data.  In other words, it searches large volumes of data for actionable ideas.

ML is a way for computers to learn about the world, much like humans use physics to learn about the world.  At the end of the day, data science is used to find patterns in data to find monentizable insights.

### Note: Back to Statistics
Another difference is that statistical modeling often uses parametric models (though not always) which aim to find causality, while machine learning models often uses non-parametric models which only aim to do prediction, and this only find correlations in the variables.  For businesses, knowing causation is less important than simply predicting.  Again, the line between statistics and machine learning is blurry, and only useful to explain buzzwords.

## Conceptual Questions on Machine Learning
1. How would you define Machine Learning?
  * Machine learning is known variables to predict some unknown variables. It's about predicting new stuff using the stuff you already know.
2. Can you name four types of problems where it shines?
  * answer
3. What is a labeled training set?
  * answer
4. What are the two most common supervised tasks?
  * answer
5. Can you name four common unsupervised tasks?
  * answer
6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?
  * answer
7. What type of algorithm would you use to segment your customers into multiple groups?
  * K-means clustering would split customers into multiple groups which are similar in-group and different between groups. One challenge is you must specify the number of groups before running the algorithm - it may be hard to say how many groups exist in the customers.
8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?
  * Spam detection is supervised learning because there is a variable to be predicted (the target variable). In unsupervised learning, there is no one variable to be predicted; the goal is to find structure in the data.
9. What is an online learning system?
  * answer
10. What is out-of-core learning?
  * answer
11. What type of learning algorithm relies on a similarity measure to make predictions?
  * answer
12. What is the difference between a model parameter and a learning algorithm's hyperparameter?
  * answer
13. What do model-based learning algorithm search for?  What is the most common strategy they use to succeed?  How do they make predictions?
  * answer
14. Can you name four of the main challenges in Machine Learning?
  * answer
15. If your model performs great on the training data but generalizes poorly to new instances, what is happening?  Can you name three possible solutions?
  1. model could be underfitting - try a more flexible model. E.g. add more variables in a linear regression, or add more hidden layers to a neural network.
  2. model could be overfitting, where the model is flexible and memorizes the training examples well, but generalizes poorly on new unseen data. The solution is to try a less flexible model (e.g. increase regularization term).
  3. Obtaining more training data can also reduce overfitting.

16. What is a test set and why would you want to use it?
  * The data is split into training and test sets. The model is trained on the training set then the model makes predictions on the test set, which it has not seen before. Test sets predict how the model will perform on new, unseen data. 

[^1]: [*Hands-On Machine Learning with Scikit-Learn & TensorFlow*](http://shop.oreilly.com/product/0636920052289.do)
