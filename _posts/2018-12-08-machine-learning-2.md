---
Title: Some Common Algorithms
Date: 2018-12-08 22:10
Author: Robert Hatem
Lang: en
Tags:
Summary: Tutorial on common statistical modeling techniques
Image: images/CRA.img
mathjax: true
---

A few more common algorithms.

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

* The gradient equals the negative of the residuals results from using the squared loss function, but we can use any loss function and derive a new gradient boosting algorithm. This result seems, that we're fitting the residuals, since we were motivated by boosting but unnecessary for our *functional gradient descent* algorithm. We can interpret it as fitting the residuals at each step, but that is a nice side-effect and not integral to the procedure.

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

See: [*XGBoost documentation: Introduction to Boosted Trees*](https://xgboost.readthedocs.io/en/latest/){:target="_blank"}
