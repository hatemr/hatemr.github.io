---
title: Model Risk Management
date: 2018-09-14
image: _images/park_ave.png
---

Mathematical models are inherently imperfect representations of economic and financial behavior.  When decisions are made based on model outputs which are incorrect or misused, that poses a risk to the firm.  To manage this risk, Model risk management is responsible for identifying, monitoring, and mitigating model risk within the firm.  The main regulatory guidance on model risk as called [FR11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm).

### Stress Testing
Stress testing arose as an ad-hoc measure to reassure markets that firms were sufficiently capitalized during the financial crisis.  After the crisis, banks continue to stress test due to requirements from regulators.

Stress testing, using a top-down approach, consists of estimating the *risk parameters* as a funcion of macroeconomic variables:
* PD - Probability of default
* LGD - Loss Given Default
* EAD - Exposure at Default

Here's my photo: ![Park Ave](/_posts/_images/park_ave.jpg)
![jpg]({filename}/_posts_images/park_ave.jpg)

Here's my photo: ![Park Ave](/images/park_ave.jpg)

## Model Validation
Model validation ensures that the stress testing model is properly developed and used.  There are a few main aspects of model validation, as laid out in [FR11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm):
1. **Evaluate Conceptual Soundness of Model Specification** - This element involves assessing the quality of the model design and construction, as well as review of documentation and empirical evidence supporting the methods used and variables selected for the model.
  * **Methods** - The *model specification* must include the correct variables, and their coefficients must make sense from an economic perspective.
* 
* **Reasonableness of Assumptions and reliability of inputs**
* 

```python
x_test = np.linspace(-10, 10, 200)
```