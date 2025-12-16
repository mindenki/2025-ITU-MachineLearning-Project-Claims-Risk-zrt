import numpy as np


# class to store info about the tree nodes
class flor_de_liz:
    def __init__(self, feature=None, threshold=None, left=None, right=None, values=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.values = values
        pass


    # checks if it is a leaf node     
    def check(self):
        return self.values is not None



# class to make the decision tree
class Decision_tree:
    def __init__(self, max_depth = None, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        pass


    def giniimp_calc(self, classes):
        unique, counts = np.unique(classes, return_counts=True)
        probabilities = counts / len(classes)
        gini = 1 - np.sum(probabilities ** 2)
        return gini


    def weighted_impurity(self, left_node, right_node, def_used):
        n_left = len(left_node)
        n_right = len(right_node)
        n_total = n_left + n_right

        weighted = (n_left / n_total) * def_used(left_node) + (n_right / n_total) * def_used(right_node)
        return weighted


    def best_split(self, features, values):
        
        n_samples, n_features = features.shape

        best_feature_indx = None
        best_threshold = None
        best_impurity = float('inf')

        # try every feature  
        for feat_ixdx in range(n_features):
            # get all unique values of the feature sorted
            feature_column = features[:, feat_ixdx]
            thresholds = np.unique(feature_column)

            # try every split between the sorted values
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2

                left_ixdx = feature_column <= threshold
                right_ixdx = feature_column > threshold

                if left_ixdx.sum() == 0 or right_ixdx.sum() == 0:
                    continue

                left_y = values[left_ixdx]
                right_y = values[right_ixdx]

                impurity = self.weighted_impurity(left_y, right_y, self.giniimp_calc)

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature_indx = feat_ixdx
                    best_threshold = threshold

        return best_feature_indx, best_threshold



    # checks the most common label in y/node 
    def most_common_label(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]



    # will be use to get predicted values for the test data
    def predict(self, food_for_prediction):
        return np.array([self.space_wagon(x, self.root) for x in food_for_prediction])
    


    # gets a value from the test data and searches for the leaf node it "belongs to" to make a prediction
    def space_wagon(self, x, node):
        # If we're at a leaf node, return its value (0 or 1)
        if node.check():
            return node.values
        
        # If not in leaf node, go to left or right node based on the feature 
        if x[node.feature] <= node.threshold:
            return self.space_wagon(x, node.left)
        else:
            return self.space_wagon(x, node.right)



    # the root node of the tree
    def fit(self, features, values):
        # Auto-fix transposed input
        if features.shape[0] != values.shape[0] and features.shape[1] == values.shape[0]:
            features = features.T

        if features.shape[0] != values.shape[0]:
            raise ValueError(
                f"Shape mismatch: X has {features.shape[0]} samples, y has {values.shape[0]}"
            )

        self.root = self.tree_maker(features, values, depth=0)
        return self


    def tree_maker(self, features, values, depth):

        n_samples = features.shape[0]
        n_classes = len(np.unique(values))
        
        # check if everything is within the tree parameters
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1 or n_samples < self.min_samples_split:

            # in this case creates a leaf node because the tree goes over the depth limit if the function continues 
            # or because the node already only has class within it
            # or the minimum sample  

            # if any of the parameters are not met, we predict the most likely value for the sample
            leaf_values = self.most_common_label(values)
            return flor_de_liz(values=leaf_values)
        
        # best split for the sample
        best_feature, best_threshold = self.best_split(features, values)

        # in case no better split is found, create leaf node
        if best_feature is None:
            leaf_values = self.most_common_label(values)
            return flor_de_liz(values=leaf_values)
        
        # split the dataset
        left_indx = features[:, best_feature] <= best_threshold
        right_indx = features[:, best_feature] > best_threshold

        left_features = features[left_indx]
        right_features = features[right_indx]
        left_values = values[left_indx]
        right_values = values[right_indx]

        # from the split creates 
        left_node = self.tree_maker(left_features, left_values, depth + 1)
        right_node = self.tree_maker(right_features, right_values, depth + 1)
        
        return flor_de_liz(feature=best_feature, 
                           threshold=best_threshold, 
                           left=left_node, 
                           right=right_node
        )