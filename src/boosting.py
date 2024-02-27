
import numpy as np
from sklearn.utils import resample
from tree import TreeNaN ,Tree
from multiprocessing import Pool, cpu_count
from loss_functions import LogLoss

class XGBoost:

    def __init__(
            self, 
            num_tree, 
            learning_rate, 
            reg_lambda,
            gamma,
            min_child_weight,
            max_depth,
            n_jobs=cpu_count(),
            loss_function=LogLoss
        ) -> None:

        self.num_tree = num_tree
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.loss_function = loss_function
        self.base_score = None
        self.trees = []

    def _get_base_score(self, y):
        """
        Return the initial base prediction of the boosting algorithm.
        It is an array of same dimension than y with all the values being the
        mean of y.
        """
        self.base_score = np.mean(y)
        return np.full_like(y, self.base_score)

    def fit(self, X, y):
        y_hat = self._get_base_score(y).astype(float)

        for _ in range(self.num_tree):
           # Resample X, y, and y_hat to build bootstrap samples of the training data
            X_resampled, y_resampled, y_hat_resampled = resample(X, y, y_hat)

            # Train a tree and append it to self.trees
            tree = TreeNaN(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
                loss_function=self.loss_function
            )
            tree.fit(X_resampled, y_resampled, y_hat_resampled)
            self.trees.append(tree)
            prediction = tree.predict(X).astype(float)
            if prediction is not None:
                ##  print(prediction)
                update = self.learning_rate * prediction
                y_hat += update

                # Calculate and print the loss for this iteration
                # loss = LogLoss.get_loss(y, y_hat)
            #     print("prediction is ")   
            #    # print(y_hat)   
            #     print(loss)    

        return self
    
    def predict(self, X):
        """
        Returns predictions for X
        """
        y_hat = self.base_score.astype(float)
        for tree in self.trees:
            update = tree.predict(X).astype(float)
            y_hat += update
        return y_hat

 