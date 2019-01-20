Title: Linear Regression
Date: 2018-04-25 09:02
Author: Emile Hatem
Lang: en
Tags: initial, pelican
Summary: Introduction to linear regression
Image: images/CRA.png

# Simple Linear Regression

## Introduction

This document serves to introduce linear regression from an elementary point of view. Returning to the theoretical foundations helps since linear regression serves as a building block for more advanced techniques.

## The Goal

This tutorial relies heavily on Cosma Shalizi's lectures on modern regression[^1], which are insightful, witty, and a delight to read. This tutorial is taken from his lectures and I recommend anyone who is curious to check them out. 

## Predicting a Random Variable

Suppose we have a variable $Y$ that we want to predict. $Y$ is a random variable whose value we don't know, but we must make a guess of it. The only information that we have is $Y$'s distribution - we have a sense of which values $Y$ are highly probable and which are less probable. Using the distribution, we can use a method that gives us the guess which is 'best' in some sense, even though we do not know the actual value of $Y$.

Let's call our prediction $m$. I use lower case because $m$ is a value - it's not a random variable. Our goal is to choose $m$ that is "best".

### Mean Sqauared Error

Let's define what we mean by 'best.' Ideally, we want our prediction $m$ to be close to the true value $Y$. Let's choose to measure how good our prediction is with the squared error $(Y - m)^2$. Other measures could be used, but using the squared error is traditional so we use it here. 

[comment]: <> (This is a comment, it will not be included in the HTML file. So you can't use inspect to see it)

Also, the squared error ignores the direction of the error - if we always overpredict, or always underpredict, the metric will not distinguish the two because of the 'squared' part. This feature is nice because errors of opposite signs can cancel each other out, leading to an overconfident picture of our estimate.

Since $Y$ is a random variable, so is the squared error. Therefore let's look at the value

$$\mathbb{E}\left[(Y - m)^2\right]$$

We call this value the *mean squared error*, $MSE(m)$, and we want to choose our prediction $m$ which minimizes it. Note that the $MSE$ is a function of $m$; by choosing the 'best' prediction, we will obtain the smallest $MSE$ possible. 

Now we must choose a value for our prediction $m$. If we don't know anything about $Y$ except its distribution, what should our one-number guess be? 

### Our prediction $m$

Intuitively, we might choose our prediction value to be the median of the distribution; it's a valid summary of the entire distribution, and maybe even a good summary. Alternatively, we could choose the mean because it's also a good summary. We would probably avoid setting $m$ to the maximum or minimum value because they are poor representations of the distribution as a whole. 

To help our analysis, let's expand the $MSE$ into two terms:

$$MSE(m) = \mathbb{E}[(Y - m)^2] = (\mathbb{E}[Y-m])^2 + Var[Y-m]$$

This is the *bias-variance* decomposition, one of the most important ideas in statistics. The first term, $(\mathbb{E}[Y-m])^2$, is the **bias** squared and it measures the expected error of using $m$ to estimate $Y$. Ideally, our estimate $m$ would have zero bias - the prediction expected to equal the true value of $Y$, but this won't always be true.

The **variance** term is $Var[Y-m] = Var[Y]$, which measures how spread out the distribution of $Y$ is. If $Y$ usually takes on values centered near one specific value, the variance will be low. It says no matter what we used as an estimate, if $Y$ is narrowly concentrated then our prediction will be better overall. The variance term does not depend on our estimate $m$, so it won't help us minimize the $MSE$. If we make some predictions, then increase $Var[Y]$, then some of $Y$'s distribution will move away from the predictions resulting in increased error. All else equal, having small $Var[Y]$ helps lower our mean squared error.

To choose our prediction, let's see which $m$ minimizes the squared error $MSE$. As we normally do in calculus, to find a minimum we will find the derivative of $MSE(m)$, set it to zero, then solve for $m$:

$$
\begin{align}
  \frac{dMSE(m)}{dm}    &= \frac{d}{dm} \left (Var[Y]+(\mathbb{E}[Y]-m)^2 \right) \\
   						&= 2 \left( \mathbb{E}[Y]-m \right) \left(\frac{d\mathbb{E}[Y]}{dm} - \frac{dm}{dm}   \right)  \\
   						&= -2 \left( \mathbb{E}[Y]-m \right) \\
   						&= 0
\end{align}
$$

### Result

Setting $-2 \left( \mathbb{E}[Y]-m \right)=0$ gives the result: $\mathbb{E}[Y] = \mu$, where we have replaced $m$ with its optimal value denoted $\mu$. This says that to minimize mean squared error, we should always predict the mean of $Y$. This of course assumes we can find the mean of $Y$, that is, we assume we know the distribution of $Y$ (or least its moments).

This sounds reasonable - if we want to predict a value of $Y$ and we have its distribution but no other information, we should predict the mean of $Y$. Since the mean is a common summary statistic of $Y$, it's not surprising that using it as a prediction is optimal.

## Predicting One Variable from Another Variable

Let's make the situation more interesting. Suppose we also have another random variable $X$ which is related to $Y$, and therefore can help us predict $Y$. That is, $X$ are $Y$ may be dependent: $p(Y \| X) \neq p(Y)$. Therefore, knowing $X$ could help us predict $Y$. Therefore, our prediction will be $m(X)$ where $m$ is now a function of $X$.

Now the $MSE$ is given by $\mathbb{E}[(Y - m(X))^2]$, and our goal, again, is to find the estimate $m(X)$ that minimizes the $MSE$. We can use the result we found previously by noticing that now $Y$ is distributed as $p(Y \| X)$ instead of $p(Y)$ as used before. Therefore, instead of predicting $m=\mathbb{E}[Y]$, we simply predict $\mu(x) = \mathbb{E}[Y \| X=x]$. This is still the mean of $Y$, but now under a new distribution of $Y$. The prediction $m$ changed to $\mu(x)$ to recognize that our prediction now uses the known $x$ value to help predict $Y$, so the prediction is a function of $x$.

As in the one-variable case, if we knew the distribution $p(Y \| X=x)$, we would just take the mean and use that as our prediction. However, we don't know the whole distribution $p(Y \| X=x)$; moreover, all we need is its mean $=\mathbb{E}[Y \| X=x]=\mu(x)$, which we call the **regression function**.

Note that for any two random variables, which have a joint distribution, the conditional expectation of one on the other can, in general, be a function of the second variable. For example, if $X_1$ and $X_2$ are bivariate normal, then $\mathbb{E}[X_1 \| X_2 = x_2]=\mu_1+\rho\frac{\sigma_1}{\sigma_2}(x_2-\mu_2) = f(x_2)$; the conditional expectation is a function of $x_2$.

### An Approximation

The actual relationship between $X$ and $Y$ could be very complicated, that is, the regression function $\mu(x)= \mathbb{E}[Y \| X=x]$ is unknown and potentially complex.

Since we don't know this complex function $\mu(x)$, which may not even have a nice mathematical form, we can choose to approximate it as a linear function of $X$:

$$\mu(x) = \mathbb{E}[Y|X=x] = \beta_0 + \beta_1x$$

It is usually a vast generalization to say that $\mathbb{E}[Y \| X=x]$ is a linear function of $X$. However, using such simple approximations allows for a starting point and often captures enough of the relationship to make useful models. It also does not assume the true relationship is linear - it is just approximating the true relationship with a linear one.

### Deriving the coefficients

Now the regression function $\mu(x)$ is defined by the coefficients $\beta_0$ and $\beta_1$. We will adjust the coefficients to find the function $\mu(x)$ which minimizes the $MSE$, which is a function of the coefficients:

$$
\begin{align}
  MSE(\beta_0, \beta_1) &= \mathbb{E}\left[\left(Y - \left(\beta_0 + \beta_1X \right)\right)^2\right] \\
   						&= \mathbb{E}[Y^2] - 2\beta_0\mathbb{E}[Y] - 2\beta_1\mathbb{E}[XY] + \mathbb{E}\left[\left(\beta_0+\beta_1X\right)^2 \right] \\
   						&= \mathbb{E}[Y^2] - 2\beta_0\mathbb{E}[Y] - 2\beta_1\left(Cov[X,Y] + \mathbb{E}[X]\mathbb{E}[Y]\right) + \beta_0^2 + 2\beta_1\mathbb{E}[X] + \beta_1^2\mathbb{E}[X^2] \\
   						&= \mathbb{E}[Y^2] - 2\beta_0\mathbb{E}[Y] - 2\beta_1(Cov[X,Y] + 2\beta_1\mathbb{E}[X]\mathbb{E}[Y] + \beta_0^2 + 2\beta_0\beta_1\mathbb{E}[X] + \beta_1^2Var[X] + \beta_1^2(\mathbb{E}[X])^2
\end{align}
$$

As before, we find the values of $\beta_0$ and $\beta_1$ which minimize the $MSE(\beta_0, \beta_1)$ by setting the derivatives equal to zero and solving for the betas:


$$\frac{d(MSE)}{d\beta_0} = 0 = -2\mathbb{E}[Y] + 2\beta_0 + 2\beta_1\mathbb{E}[X]$$

$$\frac{d(MSE)}{d\beta_1} = 0 = -2Cov[X,Y] - 2\mathbb{E}[X]\mathbb{E}[Y] + 2\beta_0\mathbb{E}[X] + 2\beta_1Var[X] + 2\beta_1(\mathbb{E}[X])^2$$

These are two equations with two unknowns, so they can be solved for $\beta_0$ and $\beta_1$. After some algebra, the result is:

$$\beta_0 = \mathbb{E}[Y] - \beta_1\mathbb{E}[X]$$
$$\beta_1 = \frac{Cov(X,Y)}{Var(X)}$$

Note that if $X$ and $Y$ are centered, that is they have mean zero, then the intercept $\beta_0$ is zero and line goes through the origin. For $\beta_1$, the more that $X$ and $Y$ vary together (tend to move when the other one moves), the steeper the line. As the variance of $X$ increases, $\beta_1$ goes to zero, regardless of its covariance with $Y$.

The result is the line $\beta_0 + \beta_1x$ is the **optimal regression line** or the **optimal linear predictor**. This line, defined by $\beta_0 and \beta_1$ is the **best** line we can choose that approximates the true regression function, according to the $MSE$.

###  Take-aways
1. We did not assume that the relationship between $X$ and $Y$ is actually linear. What we did was derive the optimal linear approximation to the true relationship, whatever that relationship might be (e.g. $\mu(x)=e^x$ or $\sin x$). This approximation is not necessarily a good one, and it may well be an awful one.

2. We did not assume anything about the marginal distributions of $X$ or $Y$ (just that the expectations exist). The optimal linear predictor doesn't require any stronger assumptions; $X$ and $Y$ can be any shaped distribution and these coefficents would still give the optimal regression line.

3. We made no assumptions on the specific distributions of the variables, $p(X)$ and $p(Y)$, nor about the joint distribution $p(X,Y)$. These assumptions can help make stronger inferences later on, but for now they are unnecessary for our purposes of predicting $Y$ using $X$ under a linear approximation.

4. In general, changing the distribution of $X$ will change the optimal regression line because both the $Cov(X,Y)$ and $Var(X)$ will change (and usually not cancel out each other's change).

5. There is no assumption that X *causes* Y; they just have non-zero covariance so one can be useful for predicting the other.

Note that to find the *optimal linear predictor*, we will need some information on our random variables; specifically, $\mathbb{E}[X]$, $\mathbb{E}[Y]$, $Cov[X,Y], Var[X]$. Since we don't know these a priori, we will need to estimate them from noisy data, making this task a *statistical* problem.

This thoery of optimal linear predictor will be used later when we introduce a more specific type of linear predictor - the *simple linear regression model*.

### Model specification
In a statistical model, we treat the variables as random varianbles. The *specification* of a statistical model says what the random variables are, and lays down more or less narrow restrictions on their distributions and how they relate to each other. We will discuss the specification of the *simple linear regression model* next.

### Simple Linear Regression[^2]

Now let's introduce the **simple linear regression model**, the "most basic of models that's actually useful for anything." Suppose we're trying to predict $Y$ from $X$. Here are the mode's assumptions:

#### Assumptions
1. The distribution of $X$ is arbitrary, possibly even deterministic.
2. If $X=x$, then $Y=\beta_0 + \beta_1x + \epsilon$ for some constants ("cefficient", "parameters") $\beta_0$ and $\beta_1$, and some random noise variable $\epsilon$ (random variable)
3. $\epsilon$ is a random variable with unspecified distribution, but has $\mathbb{E}\left[\epsilon \| X=x\right]=0$ (no matter what $x$ is), $Var\left[\epsilon \| X=x\right]=\sigma^2$ (no matter what $x$ is).
4. $\epsilon$ is uncorrelated across observations

All of these assumptions will need to be checked if the model is to be used. It is possible to use stronger assumptions and draw more powerful inferences, but these assumptions are strong enough already to start on inferece.

### Note on p-values
With the added assumption that $\epsilon$ follows a normal distribution, we can derive the sampling distributions of the coefficients. They come out to be Gaussian and we can easily measure their p-value to test the null hypothesis that the coefficients are 0. This step is out-of-scope for this tutorial but it's worth mentioning here.

P-values on the coefficients are tricky to interpret and many people have trouble with them. The term **statistical significance** has a specific definition and is not the same, nor does it imply, scientific or practical significance. Shalizi masterfully articulates the point:

> Statistical significance is a weird mixture of how big the coefficient is, how big a sample we've got, how much noise there is around the regression line, and how spread out the data is along the $x$ axis. This has so little to do with "significance" in ordinary language that it's pretty unfortunate we're stuck with the word; if the Ancestors had decided to say "statistically detectable" or "statistically distinguishable from 0", we might have avoided a lot of confusion.[^3]

When we reject the hypothesis that $\beta_1=0$, what we're saying is "It's really implausibly hard to fit this data with a flat line, as opposed to one with a slope." It prefer the term **statistically detectable** since it makes it clear that we can detect even very small coefficients.

## Summary

In conclusion, this tutorial showed that the optimal regression line of $Y$ on $X$ is defined by the coefficients $\beta_0$ and $\beta_1$ that we derived. These coefficients made no assumptions on the distributions of $X$ and $Y$, nor did we assume that the true relationship between $X$ and $Y$ is linear -- we only used the linear function as an approximation to the true relationship. The next steps would be to formalize the simple linear model by adding an error term, estimate the coefficeints from data and derive their distributions, expecations, and variances, assume that the error terms is normally distibuted, and show this assumption implies the least squares estimate gives the same result as the maximum likelihood estimate. If you want to find more information, or just want a good read on linear regression, I recommend continuing with Shalizi's [lectures](http://www.stat.cmu.edu/~cshalizi/mreg/15/). Thanks for tuning in.


References:

[^1]: [Lecture 1: Optimal Prediction](http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/01/lecture-01.pdf)

[^2]: [Lecture 3: Introducing Statistical Modeling](http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/03/lecture-03.pdf)

[^3]: [Lecture 8: Inference on Parameters](http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/08/lecture-08.pdf)