---
layout: post
title: Deep learning(10) - Error analysis
published: True
categories:
- Deep learning
tags:
- Machine learning
- Neural network
- Deep learning
- Python
- Tensorflow
typora-root-url: /Users/shephexd/Dropbox/github/pages/
---

After building your model quickly, how can you improve your model?

What if your model performance is lower than your expectation.

Can you know the problem your model have?

`Error analysis` is a way to address your model to improve performance.



<!--more-->

This post is based on the video lecture[^1] 



## Error Analysis



### How to find error

1. Get small sample for mislabeled dev set
2. Count up how many is correct



Error analysis is depend on the **ceiling** meant Upper bound on how much you could improve performance.



To improve performance, consider which idea is helpful.



### Evaluate multiple idea in parallel

Draw a table to evaluate your ideas



| Image      | Dog  | Cat  | Tiger | Comment |
| ---------- | ---- | ---- | ----- | ------- |
| 1          |      |      |       |         |
| 2          |      |      |       |         |
| ...        |      |      |       |         |
| % of total | 8 %  | 43%  | 61%   |         |



Find out which problem is **critical** for performance.



## Clean up incorrectly labeled data

Deep learning algorithms are quite robust to random errors in the training set. But, less robust to systematic errors.



#### Find out  problems

- Apply same process to your dev and test sets to make sure they continue to come from same distribution.

- Consider examining examples your algorithms got right as well as ones it got wrongs.
- Train and dev/test data may now come from slight different distribution.



#### Example

- Overall dev set error $\cdots\ 10\%$
- Errors due incorrect labels $\cdots \ 0.6\%$
- Errors due to other reasons $\cdots \ 9.4\%​$





## Mismatched training and dev/test set

Assuming that there are data set like the below.

- train set: 20,000 from web + 5,000 from mobile
- dev/test set: 2500 / 2500 from mobile



The problem is ...

1. The algorithm saw the data in the training set but not in the dev.
2. The distribution of data in the dev set is different from training



The data divide into train, training-dev, dev and test set.

- train set: 90%
- training-dev set: 4%
- dev set: 3%
- test set: 3%



Training-dev: Same distribution as training set, but not used for training.



### Case for errors

- Human error $\cdots\ ​$ Avoidable bias
- Training error $\cdots\ $ Avoidable bias / variance
- Training-dev error $\cdots ​$ variance / data mismatched
- Dev error $\cdots $ data mismatched / degree of overfitting to dev set
- Test error$\cdots​$ degree of overfitting to dev set



## Conclusion

Error analysis will be helpful to guide you for better performance.

The main problem might be one of them, model or data.



*Build your first system quickly, then iterate*



- carry out manual error analysis to try to understand difference between training and dev/test.
- Make training data more similar to test.
- Collect More data similar to dev/test sets.





[^1]: https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning	"Deep learning specialization"