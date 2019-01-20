---
Title: Note on Training, Validation, and Test sets
Layout: post
Date: 2018-09-14
Image: _images/park_ave.png
---

Splitting into training and test sets is easy — you want to estimate model performance on new, unseen data.  It tells you how the model would perform after implementing in production.

But what about the validation set?  Why split into a third set?  Because validation sets are used to select the hyperparameters:

1. Training set – to estimate the parameters (e.g. coefficients in linear regression model, parameters in a neural network).
2. Validation set – to estimate hyperparameters (e.g. which variables to include in a linear regression, number of hidden layers in a neural network).
3. Test set – to estimate model performance on new, unseen data.

If you don’t have any hyperparameters to select, then you don’t need a validation set; just training and test will work!