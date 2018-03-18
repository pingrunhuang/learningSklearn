
# Decision session
result of using `entropy` as criterion:
![]("decision-tree/dt-entropy.png")

result of using `gini` as criterion:
![]("decision-tree/dt-gini.png")

as shown above, using entropy will generate a over-fitting tree.

#### result statics
```
accuracy: 0.7959183673469388
classfication report:              
            precision    recall  f1-score   support

            0       0.88      0.88      0.88       376
            1       0.31      0.32      0.32        65

avg / total       0.80      0.80      0.80       441

confusion matrix: [[330  46]
 [ 44  21]]
```