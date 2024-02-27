from typing import Optional
import numpy as np
from multiprocessing import Pool, cpu_count
from loss_functions import LogLoss


class Node:
    def __init__(self, 
               feature: int, 
               value: float, 
               left: Optional['Node']=None, 
               right: Optional['Node']=None,
               nan_direction: str = 'left'):
        
        self.feature: int = feature
        self.value: float = value
        self.left: Optional[Node] = left
        self.right: Optional[Node] = right
        self.nan_direction = nan_direction


class Tree:

    def __init__(self, 
                 reg_lambda, 
                 gamma, 
                 max_depth,
                 min_child_weight, 
                 n_jobs=cpu_count(), 
                 loss_function=LogLoss):
        
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.n_jobs = n_jobs,
        self.loss_function = loss_function
        self._tree = None

    def _get_leaf_weight(self, y, y_hat):
        """
        Returns the weight for each leaf
        """
        grad = self.loss_function.get_gradient(y, y_hat)
        hess = self.loss_function.get_hessian(y, y_hat)
        return -np.sum(grad) / (np.sum(hess) + self.reg_lambda)
    def _get_score(self, grad_left, grad_right, hess_left, hess_right):
        """
        Returns the loss function of the left and right children L_L + L_R
        """
        epsilon = 1e-8  # small constant to avoid division by zero
        score_left = (grad_left ** 2 / (hess_left + self.reg_lambda + epsilon)) if hess_left != 0 else 0
        score_right = (grad_right ** 2 / (hess_right + self.reg_lambda + epsilon)) if hess_right != 0 else 0
        return score_left + score_right
    
    def _find_best_split_for_feature(self, x, y, y_hat, grad_parent, hess_parent):
        """
        Find the score and index of the best split in column x
        """
        # initially all the samples are on the right
        grad_left = 0
        hess_left = 0
        grad_right = grad_parent
        hess_right = hess_parent

        # we don't need to sort the feature, we can just find the order 
        # of the indices in the feature
        indices = np.argsort(x)

        # This is the initial guess of the best score and the related
        # index of the value 
        best_score = 0
        best_idx = 0

        # TODO: iterate through the indices and test the different splits for
        # each indice. Compute the new score and replace the best_score 
        # best best_idx if the new score is greater.
        for idx in indices:

                g = self.loss_function.get_gradient(y[idx], y_hat[idx])
                h = self.loss_function.get_gradient(y[idx], y_hat[idx])
                grad_left += g
                hess_left += h
                grad_right -= g
                hess_right -=h
                score = self._get_score(grad_left, grad_right, hess_left, hess_right)
                if score > best_score:
                        best_score = score
                        best_idx = idx


        return best_score, best_idx

    def _find_best_split(self, X, y, y_hat):
        """
        Find the score, index, and feature of the best split in data X
        """
        best_score = 0
        best_idx = 0
        best_feature = 0

        # TODO: compute the initial value for grad_parent and hess_parent
        grad_parent = self.loss_function.get_gradient(y, y_hat).sum()
        hess_parent = self.loss_function.get_hessian(y, y_hat).sum()

        for feature in range(X.shape[1]):
            x = X[:, feature]
            score, idx = self._find_best_split_for_feature(x, y, y_hat, grad_parent, hess_parent)

            if score > best_score:
                best_score = score
                best_idx = idx
                best_feature = feature
  
        # We substract the loss of the parent node so we can compare
        # the score to the gamma parameter
        best_score -= grad_parent ** 2 / (hess_parent + self.reg_lambda)
        return best_feature, best_idx, best_score
    
    def _find_best_split_parallel(self, X, y, y_hat):
        """
        Find the score, index, and feature of the best split in data X 
        using multiple threads
        """
        best_score = 0
        best_idx = 0
        best_feature = 0

        grad_parent = self.loss_function.get_gradient(y, y_hat).sum()
        hess_parent = self.loss_function.get_hessian(y, y_hat).sum()

        args_list = [
            (feature, y, y_hat, X, grad_parent, hess_parent)
            for feature in range(X.shape[1])
        ]

        def get_best_split_for_feature(args):
            """
            Inner function to compute the best split for a specific feature
            """
            feature, y, y_hat, X, grad, hess = args

            # TODO: compute best_score and best_idx for feature
            feature, y, y_hat, x, grad, hess = args
            best_score, best_idx = self._find_best_split_for_feature(x, y, y_hat, grad, hess)
            return feature, best_score, best_idx

        with Pool(self.n_jobs) as pool:
            results = pool.map(get_best_split_for_feature, args_list)

        for feature, score, idx in results:
            if score > best_score:
                best_score = score
                best_idx = idx
                best_feature = feature

        # We substract the loss of the parent node so we can compare
        # the score to the gamma parameter
        best_score -= grad_parent ** 2 / (hess_parent + self.reg_lambda)
        return best_feature, best_idx, best_score

    def _build_tree(self, X, y, y_hat, depth):
        """
        Recursivelly build the tree

        - We don't create a new node if the shape the data is 0 
        - We return a leaf node if the total Hessian measured on the data
        is smaller or equal than min_child_weight
        - We return a leaf node if the depth is greater or equal to max_depth
        - We return a leaf node if the score of the best split is smaller than gamma
        - We return an inner node otherwise    
        """
        if X.shape[0] == 0:
            return None
        total_hessian = self.loss_function.get_hessian(y, y_hat).sum()
        if total_hessian <= self.min_child_weight:
            return Node(feature=None, value=self._get_leaf_weight(y, y_hat))

        if depth >= self.max_depth:
            return Node(feature=None, value=self._get_leaf_weight(y, y_hat))

        feature, idx, score = self._find_best_split(X, y, y_hat)

        if score < 2 * self.gamma:
            return Node(feature=None, value=self._get_leaf_weight(y, y_hat))
        left_mask = X[:, feature] <= X[idx, feature]
        right_mask =  X[:, feature] >= X[idx, feature]
        left_node = self._build_tree(X[left_mask], y[left_mask], y_hat[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], y_hat[right_mask], depth + 1)
        return Node(feature=feature, value=X[idx, feature], left=left_node, right=right_node)

    def _find_prediction(self, sample, node):
        """
        Return the prediction for the sample
        """
        # TODO: recursively iterate through the tree 
        # to find the prediction for the sample
        if node and node.left is None and node.right is None:
            return node.value

        if node and sample[node.feature] <= node.value:
            return self._find_prediction(sample, node.left)
        if node:
            return self._find_prediction(sample, node.right)
    
    def fit(self, X, y, y_hat):
        self._tree = self._build_tree(X, y, y_hat, 1)
        return self
    
    def predict(self, X):
        predictions = [self._find_prediction(sample, self._tree) for sample in X]
        return np.array(predictions)
    

class TreeNaN(Tree):

    def _find_best_split_for_feature(self, x, y, y_hat, grad_parent, hess_parent):

        non_nan_indices = np.where(~np.isnan(x))[0]
        nan_indices = np.where(np.isnan(x))[0]
        sorted_indices = non_nan_indices[np.argsort(x[non_nan_indices])]

        best_score = 0
        best_idx = 0  
        best_nan_direction = 'left'
        grad_nan = self.loss_function.get_gradient(y[nan_indices], y_hat[nan_indices]).sum()
        hess_nan = self.loss_function.get_hessian(y[nan_indices], y_hat[nan_indices]).sum()

        grad_left = 0
        hess_left = 0
        grad_right = grad_parent - grad_nan
        hess_right = hess_parent - hess_nan
        for i in range(len(sorted_indices)):
            idx = sorted_indices[i]
            if i > 0 and x[idx] == x[sorted_indices[i - 1]]:
                continue

            g = self.loss_function.get_gradient(y[idx], y_hat[idx])
            h = self.loss_function.get_hessian(y[idx], y_hat[idx])

            grad_left += g
            hess_left += h
            grad_right -= g
            hess_right -= h

            score = self._get_score(grad_left, grad_right, hess_left, hess_right)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_nan_direction = 'left'
                best_grad_left = grad_left
                best_hess_left = hess_left
                best_grad_right = grad_right
                best_hess_right = hess_right

        score_nan_left = self._get_score(best_grad_left + grad_nan, best_grad_right, best_hess_left + hess_nan, best_hess_right)
        score_nan_right = self._get_score(best_grad_left, best_grad_right + grad_nan, best_hess_left, best_hess_right + hess_nan)

        if score_nan_left > best_score:
            best_score = score_nan_left
            best_idx = nan_indices[0]
            best_nan_direction = 'left'

        if score_nan_right > best_score:
            best_score = score_nan_right
            best_idx = nan_indices[0]
            best_nan_direction = 'right'

        # Return best split info, including NaN handling direction
        return best_score, best_idx, best_nan_direction
    
    def _find_best_split(self, X, y, y_hat):
        best_score = 0
        best_idx = 0
        best_feature = 0

        grad_parent = self.loss_function.get_gradient(y, y_hat).sum()
        hess_parent = self.loss_function.get_hessian(y, y_hat).sum()

        for feature in range(X.shape[1]):
            x = X[:, feature]
            score, idx, nan_direction = self._find_best_split_for_feature(x, y, y_hat, grad_parent, hess_parent)

            if score > best_score:
                best_score = score
                best_idx = idx
                best_feature = feature

        best_score -= grad_parent ** 2 / (hess_parent + self.reg_lambda)
        return best_feature, best_idx, best_score
    
    def _find_best_split_parallel(self, X, y, y_hat):
        # TODO: implement
        raise NotImplemented
    
    def _build_tree(self, X, y, y_hat, depth):
        if X.shape[0] == 0:
            return None

        if depth >= self.max_depth:
            return Node(feature=None, value=self._get_leaf_weight(y, y_hat))

        feature, idx, score = self._find_best_split(X, y, y_hat)

        if score < 2 * self.gamma:
            return Node(feature=None, value=self._get_leaf_weight(y, y_hat))

        left_mask = X[:, feature] <= X[idx, feature]
        right_mask = X[:, feature] >= X[idx, feature]
        left_node = self._build_tree(X[left_mask], y[left_mask], y_hat[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], y_hat[right_mask], depth + 1)
        return Node(feature=feature, value=X[idx, feature], left=left_node, right=right_node)
    
    def _find_prediction(self, sample, node):
        if node is None:
            return 0
        
        if np.isnan(sample[node.feature]).any():
            if node.left is not None:
                return self._find_prediction(sample, node.left)
            elif node.right is not None:
                return self._find_prediction(sample, node.right)
        
        if node.left is None and node.right is None:
            return node.value

        if sample[node.feature] <= node.value:
            return self._find_prediction(sample, node.left)
        else:
            return self._find_prediction(sample, node.right)


