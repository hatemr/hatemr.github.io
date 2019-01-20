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

# Overview of  Machine Learning
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

For example, linear regression can be regularized using the L1 norm (Lasso) and L2 norm (Ridge) on the cost function, limiting the possible values the coefficients can take on.  This will raise the model's bias, but lower it's variance, hopefully reducing the test error.  The amount of regularization is adjusted to minimize the test error.

Neural networks are highly flexible models which must be carefully checked, and regularized, to avoid overfitting.  Though their flexibility makes them diffciult to understand, like a black-box, when appropriately regularized and applied to the correct problems neural networks offer highly accurate predictions.


### Model Validation: Making Predictions, not Economic Theory
In banking and finance, the terminology largely comes from econometrics.  For example, they refer to the mathemtical model and the predictor variables as the __model specification__.  Historically, the economic theory chose the predictor variables, and trying different combinations to find the best fit was scorned as _data mining_.  The variables had to make good economic sense, based on economic theory.  However,  most economists now recognize that instead of always starting from theory, empirical data can reveal economic theory, and trying many different model specifications has gained acceptance.

Choosing the right model specification is emphasized far less in the machine learning community.  The usual approach is to try many combinations of variables, with many mathematical models, to find the best fit.  Accidentally omitting a variable could lower predictive accuracy, but it generally doesn't violate any theory or laws like it might in economics.  For example, in image recognition there is no notion that the pixel at position (XX, YY) should have a certain relationship with the target variable (cat or no cat).  It does not violate any laws, and as long as the model predicts with high accuracy, the specific model is not particularly important.

In econometrics, using models that have been checked and tested for a correct specification allows one to make strong statements about the relationships between the variables, particularly, causality.  In short, the extra works allows one to find new economic theory and laws (to the extent that it actually exist).  Returning to image recognition, people do not use their model to form a theory about which speicifc arraingment of pixels defines  our notion of a "cat."  No law was discovered; but it's used powerfully in applications, like self-driving cars.

In a business setting, models are used mainly for prediction, not theory-building.  Therefore, rigorously checking assumptions for a "correct specification" is often uneeded.  It is not important to checking that the correct set of variables was used; as long as the model predicts accuractely, then the set is not very important.  Checking the variables, and their coefficient signs, can help point towards a more predictive set, but they need not be perfect.

The guidance on model validation, namely SR11-7, adopts the mindset and terminology of an academic econometrician who is focused on carefully finding economics theory, not on practical business predictions.  This imposes overly strict requirements on the models, which do not correspond to the reality of their usage.  This mismatch between strict requirement and actual usage is not surprising given that some level of arbitrariness is always present when writing and fulfilling regulation, but a more nuansced approach could be considered.


## Is Linear Regression part of Machine Learning?
Yes!  Linear regression is a technique in machine learning.  The term *machine learning* is basically a buzzword for statistics.  Machine learning is basically the computer science community's word for statistics.  It's just a buzzword.  For someone who is unconvinced, notice that linear regression is just a single-layer neural network with the identity activation function; if deep neural networks count as machine learning then so does linear regression.

### Buzzwords
There are a lot of buzzwords flying around; it can be hard to know what it all means.  Here are some common buzzwords and what they actually mean. 
* Artificial Intelligence - Machine achieving human-level performance at some specific task (face recognition, credit approval, recognizing speech)

* Machine Learning - Subset of AI that teaches a computer to perform a task from experience (i.e. data), rather than having it pre-programmed beforehand by a human.

* Data Mining - Uses ML to find patterns in data in a quest for actionable ideas.

* Big Data - Data mining on large sets of structured and unstrauctured (text, speech) data

* Data Science - Science of performing data mining on Big Data.  In other words, it searches large volumes of data for actionable ideas.

ML is a way for computers to learn about the world, much like humans use physics to learn about the world.  At the end of the day, data science is used to find patterns in data to find monetizable insights.

### Note: Back to Statistics
Another difference is that statistical modeling often uses parametric models (though not always) which aim to find causality, while machine learning models often uses non-parametric models which only aim to do predition, and this only find correlations in the variables.  For businesses, knowing causation is less improtant than simply predicting.  Again, the line between statistics and machine learning is blurry, and only useful to explain buzzwords.

## Conceptual Questions on Machine Learning
1. How would you define Machine Learning?
  * answer
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
  * answer
8. Would you frame the problem of spam detection as a suptervised learning problem or an unsupervised learning problem?
  * answer
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
15. If your model performs great on the training data but generalizes poorly to new instances, what is happending?  Can you name three possible solutions?
  * answer
16. What is a test set and why would you want to use it?
  * answer

[^1]: [*Hands-On Machine Learning with Scikit-Learn & TensorFlow*](http://shop.oreilly.com/product/0636920052289.do)


## Decision Trees
Decision trees split the feature space into non-overlapping regions, then predict the average value in the region (regression), or probability of being in any given class based on fraction of each class  in the region (classification). The features space is split into regions through recursive binary splitting. At each step, a region is split in two based on whichever feature and splitting threshold minimizes the objective function. The objective function is usually Gini impurity or entropy. For each split, the feature and threshold is found which will cause the largest decrease in impurity. This requires searching through all features and possible thresholds for each split. The algorithm continues splitting until some stopping criteria is met (e.g. maximum depth is reached).

## Boosting
The boosting approach is to fit a weak learners sequestially to weighted versions of the data, where more weight is given to examples that were misclassified by earlier rounds. On its face this approach sounds like cheating; it's like a teacher allowing you to re-take a test only redoing the questions you got wrong the first time. With this interpretation the idea is quite intuitive. What other grade school hacks could inspire new learning algorithms...

## Combining Classifiers
Classifying a point by a decision tree can be seen as a sequence of classifiers, refined as we follow the path to a leaf.

A more general formulation is to combine classifiers $h_1(\mathbf{x}),...,h_m(\mathbf{x})$ via a weighted sum:

$$ H(\mathbf{x}) = \alpha_1 h_1(\mathbf{x}) +\dots + \alpha_m h_m(\mathbf{x}) $$

where the weight $\alpha_j$ is the vote assigned to classifier $h_j$. These classifiers can be simple (e.g. based on a single feature). Votes should be higher for more "reliable" classifiers. We then predict based on the sign of this score:

$$ \hat{y}(\mathbf{x}) = \text{sign} \ H(\mathbf{x}) $$

To train such a model, we cannot search the whole space of possible classifiers to form the final classifier $H(\mathbf{x})$. Instead, we use a gready algorithm to assemble the individual classifiers one-at-a-time.

Consider a family of classifiers $\mathcal{H}$ parameterized by $\theta$. To set the first classifier, we set $\theta_1$ by minimizing the training error

$$ \sum_{i=1}^N L(h(\mathbf{x_i} ;\theta_1), y_i) $$

where $L$ is some surrogate for the 0/1 loss. We then give this classifer $h_(\mathbf{x}, \theta_1)$ a weight $\alpha_1$ and set the total classifier to be $H(\mathbf{x}) = \text{sign} (\alpha_1 h(\mathbf{x} ,\theta_1))$. We now add a second classifer $ h(\mathbf{x} ,\theta_2)$, but how do we set $\theta_2$? We minimize the (surrogate) loss of the updated classifer:

$$ \sum_{i=1}^N L(H(\mathbf{x_i}), y_i) $$

where we added in the second classifier; $H(\mathbf{x}) = \text{sign}( \alpha_1 h(\mathbf{x} ,\theta_1)+ \alpha_2 h(\mathbf{x} ,\theta_2))$.

This section relies heavily on the notes from my class at the Toyota Technological Institute at Chicago, taught be Prof. Greg Shakhnarovich. The notes are not available freely online but the class can be found here: [*TTIC 31020: Introduction to Machine Learning (Shakhnarovich)*](http://shop.oreilly.com/product/0636920052289.dhttps://xgboost.readthedocs.io/en/latest/tutorials/model.htmlohttp://ttic.uchicago.edu/~gregory/#teaching)

## AdaBoost
Adaboost uses boosting, fitting a series of model sequentially on weighted versions of the dataset. The weights focus the model on the data points that it misclassified in the previous steps:

**Algorithm**:
Greedy algorithm for $m=1,...,M$:
* Initial weights $W_i^{(m)}$ to all $1/N$. These are weights for each data point $i$ at iteration $m$.
* Pick a weak classifier $h_m$ minimizing error $\epsilon_m$, weighting each data point by $W^{(m-1)}$.
* Set the weight to the classifier as $\alpha_m = \frac{1}{2} \log \frac{1-\epsilon_m}{\epsilon_m}$. This gives high weight to models with low error $\epsilon_m$.
* Update the weights on the training points $W_i^{(m)}$, based on the mistakes by model $h_m$ and the model's weight $\alpha_m$. 
* The final (strong) classifier is $ H(\mathbf{x}) = \text{sign} (\sum_m \alpha_m h_m(.)  ) $



## Gradient Boosting
Gradient boosting is the most successful method in Kaggle competitions. It cleanly handles continuous and categorical variables, handles missing data, and is harder to mess up than other powerful techniques like deep learning. The model is not new; it's roots trace to a paper by Friedman in 2001, but become useful only recently with the **XGBoost** library. A large fraction of recent Kaggle competition winners have used gradient boosting; to rank highly, your best bet is to start with gradient boosting, even though if that  might not be great science.

In boosting, we build an additive model where each model fits the residuals from the previous model. A slight variation applies in gradient boosting. In boosting, to fit the residuals we find the parameters of a parametric function which minimize the (squared) loss:

$$ \theta_1 = \underset{\theta}{\arg\min} \sum_{i=1}^N (y_i - f(x_i; \theta) )^2 $$

In **gradient boosting**, we rewrite the loss as a function of the target variable and predictor variables:

$$ L(\mathbf{Y}, \mathbf{X}) = \frac{1}{2} \sum_{i=1}^N (y_i - F(\mathbf{x}_i) )^2 $$

Usually we perform gradient descent on the parameters $\theta$ of the function $F(\mathbf{x_i}; \theta)$. Instead, we treat the function applied at each data point $F(\mathbf{x_1}),...,F(\mathbf{x_N})$ as the parameters of $L$.

Now take the gradient of the loss with respect to these new parameters:

$$
\frac{\partial L(\mathbf{Y}, \mathbf{X};F) }{\partial F(\mathbf{x_i})} = F(\mathbf{x_i}) - y_i
$$

So the residuals $y_i - F(\mathbf{x_i})$ are the negative of the gradient of the loss, when we treat the funtion $F(\mathbf{x_i})$ as the parameters. The gradient will be $N$-dimensional since there are $N$ data points.

We show the gradient boosting algorithm:

* Start with an initial model, say, the best fit of a constant function:
$$ F_1(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N y_i $$
* for $m=1,...$ until convergence:
    * Calculate the negative gradient for each data point
    $$ -g(\mathbf{x_i}) = -\frac{\partial L(y_i, F_m(\mathbf{x_i})) }{\partial F_m(\mathbf{x_i})}  =  y_i - F(\mathbf{x_i}) $$
    This is the gradient evaluated at $F_m$ (the current prediction function), at the $N$ data points.
    * Fit (least squares) a regression function $f_{m+1}$ to the negative gradients
    $$ f_{m+1} \approx -g(\mathbf{x_i}) $$
    This fits the residuals from the previous step since the residuals equal the negative gradient.
    * Update the model by making a step in the direction of the negative gradient
    $$ F_{m+1} = F_m + \eta f_{m+1}  $$
    Now adding the next function is our additive model is equivalent to taking a gradient step in functional space.

### Gradient descent in functional space
Notice in the algorithm:

* Typically, we perform gradient descent on parameters to minimize a specific loss function, giving the best point in parameter space. Now, in gradient boosting, we are doing functional gradient descent; find the best point in functional space. At each step, we are taking a step by adding a *function*.

* It is difficult to picture what functional space is, but it's easy to see the descent as taking a step in functional space: $F_{m+1}=F_m + \eta f_{m+1}$

* At each step, we want to move the master function in the direction of the negative gradient. Instead of directly calculating the gradient in functional space (I'm not sure what that would even mean), we take $N$ samples of the gradient, then find a function which best fits these samples, and you that as our best approximation of the gradient. We could simply step in the direction of the gradient, but it only optimizes the function $F$ at a fixed set of $N$ points, so we do not learn a function that can generalize to new data. 

* The gradient equally the negative of the residuals results from using the squared loss function, but we can use any loss function and derive a new gradient boosting algorithm. This nice result, we're fitting the residuals, seems nice since we were motivated by boosting but unnecessary for our *functional gradient descent* algorithm. We can interpret it as fitting the residuals at each step, but that is a nice side-effect and not integral to the procedure.

* Gradient boosting is general; all we need is a (sufficiently) differentiable loss function. For example, the absolute error loss function is $ y_i - f(\mathbf{x_i})$ and it's derivative with respect to $f(\mathbf{x_i})$ is $\text{sign}(y_i - f(\mathbf{x_i}))$

### Gradient boosting for classification
Before we looked at regression but now let's look classification with $C$ classes. Clearly this is just an example of using a different loss function and weak learner (tree instead of linear regression).

Now the prediction function $F$ predicts a *score matrix* in $R^{N \times C}$. Each row is scores for each of the $C$ classes for one data point. The columns are the scores across data points for one class. These scores can be converted to probabilities by normalizing each row

$$ p(y_i=c|\mathbf{x_i}) = e^{F_c{\mathbf{x_i}}} / \sum_k {F_k{\mathbf{x_i}}} $$

As defined, the negative gradients are also a $N \times C$ matrix, with entry at $(c,i)$
$$ -g_c(\mathbf{x_i}) = - \frac{\partial L}{\partial F_c(\mathbf{x_i})} $$

The loss function will have a different form than in regression, so the gradient will too. As before, the gradient boosting algorithm is to start with an function for each class $F_1^{(1)},...,F_C^{(1)} $ and at each iteration $m$ fit the next function for each class $f_c^{(m+1)}$ to the $N$ negative gradients $ -g_c(\mathbf{x_1}),...,-g_c(\mathbf{x_N})$. Clearly the function will not be a linear regression but some other weak learner, such as decision trees.

### Step size
The update step in gradient boosting is
$$ F_{m+1} = F_m + \eta f_{m+1} $$

where $\eta$ is the step size. How do we choose $\eta$?

An aggressive approach is the to choose the $\eta$ which would minimize the loss function if we were to use it:
$$ \eta_m = \underset{\eta}{\arg\min} \sum_{i=1}^N L(y_i, F_m(\mathbf{x_i}+\eta )+F_{m+1}(\mathbf{x_i})) $$

Other approaches are to use a fixed value $\eta < 1$ or a decaying $\eta_m$.


### XGBoost
XGBoost is a highly useful implementation of gradient boosting. It uses CART trees as the weak learners and can use any differentiable loss function.

See: [*XGBoost documentation: Introduction to Boosted Trees*](http://shop.oreilly.com/product/0636920052289.dhttps://xgboost.readthedocs.io/en/latest/tutorials/model.htmlo)