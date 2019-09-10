---
Title: Empirical risk minimization uses a Monte Carlo approximation
Date: 2019-09-10
Author: Robert Hatem
Lang: en
Tags:
Image: images/life_expectancy.png
mathjax: true
---

In machine learning, we often choose the model $\delta$ which minimizes the expected _risk_. The risk is the expected loss of the model over the data:

$$ R(p_{*}, \delta) = \mathbb{E}_{(\bf{x}, \mathit{y}) \sim p_{*} } [L(y, \delta \bf(x))]  $$ 

However, we usually don't know the true data-generating distribution $p_{*}$. So we must use a __Monte Carlo__ to approximate this expectation (i.e. integral). 

First, we approximate the distribution of $L(y, \delta \bf(x))$ with the _empirical_ distribution $$ \{L(y_i, \delta \bf(x_i) \}_{i=1}^N $$. We simply draw samples, then compute the arithmetic mean of the function applied to the samples:

$$ R_{emp} = R(p_{emp}, \delta) = \frac{1}{N} \sum_{i=1}^{N} L(y_i, \delta(\bf{x}_i))
$$

This is standard empricical risk minimization (ERM), but I want to point out that the emprical risk $R_{emp}$ is a Monte Carlo approximation of an integral. 

Recall that in Monte Carlo approximations, we approximate the mean (or other integral) of some statistic using finite samples:

$$ \mathbb{E}(f(x)) = \int f(x)p(x)dx \approx \frac{1}{N} \sum_{i=1}^N f(x_i)
$$

Monte Carlo approximation has the advantage over numerical integration (which is based on evaluating the function at a fixed grid of points) that the function is only evaluated in places where there is non-negligible probability (Murphy p.53). This explains why Monte Carlo is used to approximate integrals and not other numerical integration methods.


## References
* Machine Learning from a Probabilistic Perspective (Murphy), p. 204-205
* Natural Language Understanding with Distibuted Representations (Cho). p. 8-9