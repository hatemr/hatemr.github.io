---
Title: Model Risk Management
Date: 2018-09-14
Image: images/park_ave.png
---

Mathematical models are inherently imperfect representations of economic and financial behavior.  When decisions are made based on model outputs which are incorrect or misused, that poses a risk to the firm.  To manage this risk, Model risk management is responsible for identifying, monitoring, and mitigating model risk within the firm.  The main regulatory guidance on model risk as called [FR11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm).

# Make predictions, not economic theory

In banking and finance, the terminology largely comes from econometrics. For example, they refer to the mathematical model and the predictor variables as the model specification. Historically, the economic theory chose the predictor variables and trying different combinations to find the best fit was scorned as data mining. The variables had to make good economic sense, based on economic theory. However, most economists now recognize that instead of always starting from theory, empirical data can reveal economic theory, and trying many different model specifications has gained acceptance.

Choosing the right model specification is emphasized far less in the machine learning community. The usual approach is to try many combinations of variables, with many mathematical models, to find the best fit. Accidentally omitting a variable could lower predictive accuracy, but it generally doesn’t violate any theory or laws like it might in economics. For example, in image recognition, there is no notion that the pixel at position (XX, YY) should have a certain relationship with the target variable (cat or no cat). It does not violate any laws, and as long as the model predicts with high accuracy, the specific model is not particularly important.

In econometrics, using models that have been checked and tested for a correct specification allows one to make strong statements about the relationships between the variables, particularly, causality. In short, the extra works allows one to find new economic theory and laws (to the extent that it actually exist). Returning to image recognition, people do not use their model to form a theory about which specific arrangement of pixels defines our notion of a “cat.” No law was discovered; but it’s used powerfully in applications, like self-driving cars.

In a business setting, models are used mainly for prediction, not theory-building. Therefore, rigorously checking assumptions for a “correct specification” is often unneeded. It is not important to check that the correct set of variables was used; as long as the model predicts accurately, then the set is not very important. Checking the variables, and their coefficient signs, can help point towards a more predictive set, but they need not be perfect.

The guidance on model validation, namely SR11-7, adopts the mindset and terminology of an academic econometrician who is focused on carefully finding economics theory, not on practical business predictions. This imposes overly strict requirements on the models, which do not correspond to the reality of their usage. This mismatch is not surprising; regulation can’t anticipate everything, but a more nuanced approach could be considered.