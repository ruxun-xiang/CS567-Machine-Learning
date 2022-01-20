import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY CODES ABOVE
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    real_labels = np.array(real_labels)
    predicted_labels = np.array(predicted_labels)
    tp = np.sum(real_labels * predicted_labels, axis=0)
    fp = np.sum((1 - real_labels) * predicted_labels, axis=0)
    fn = np.sum(real_labels * (1 - predicted_labels), axis=0)

    if tp + fp == 0:
        p = 0
    else:
        p = tp / (tp + fp)
    if tp + fn == 0:
        r = 0
    else:
        r = tp / (tp + fn)
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2.0 * p * r / (p + r)
    return f1


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.sum(np.power(np.abs(point1 - point2), 3)) ** (1 / 3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.sqrt(np.sum(np.dot((point1 - point2), (point1 - point2))))

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """

        point1 = np.array(point1)
        point2 = np.array(point2)
        num = np.dot(point1, point2)
        denom = np.linalg.norm(point1) * np.linalg.norm(point2)
        if denom != 0:
            cos_sim = num / denom
        else:
            cos_sim = 0
        return 1.0 - cos_sim


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of
        k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you
        need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        best_f1 = -1

        best_param = {}

        for k in range(1, 30, 2):
            for name, func in distance_funcs.items():
                model = KNN(k, func)
                model.train(x_train, y_train)
                y_predict = model.predict(x_val)
                f1 = f1_score(y_val, y_predict)

                if f1 > best_f1:
                    best_f1 = f1
                    best_param["f1"] = f1
                    best_param["func_name"] = name
                    best_param["k"] = k
                    best_param["model"] = model

        self.best_k = best_param["k"]
        self.best_distance_function = best_param["func_name"]
        self.best_model = best_param["model"]




    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

        best_f1 = -1
        best_param = {}
        for scaler_name, scaler_class in scaling_classes.items():
            scaler_func = scaler_class()
            scaled_xt = scaler_func(x_train)
            scaled_xv = scaler_func(x_val)
            for k in range(1, 30, 2):
                for dist_name, dist_func in distance_funcs.items():
                    model = KNN(k, dist_func)
                    model.train(scaled_xt, y_train)
                    y_pred = model.predict(scaled_xv)
                    f1 = f1_score(y_val, y_pred)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_param["f1"] = f1
                        best_param["func_name"] = dist_name
                        best_param["k"] = k
                        best_param["model"] = model
                        best_param["scaler"] = scaler_name

        self.best_k = best_param["k"]
        self.best_distance_function = best_param["func_name"]
        self.best_model = best_param["model"]
        self.best_scaler = best_param["scaler"]



class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized_features = []
        for i in range(len(features)):
            feature = np.array(features[i])
            norm = np.linalg.norm(feature)
            if norm != 0:
                normalized_features.append(np.array(features[i] / norm).tolist())
            else:
                normalized_features.append(features[i])
        return normalized_features

        # normalized_features = [0 * len(features[0])] * len(features)
        # for i in range(len(features)):
        #     feature = np.array(features[i])
        #     norm = np.linalg.norm(feature)
        #     if norm != 0:
        #         normalized_features[i] = np.array(features[i] / norm).tolist()
        #     else:
        #         normalized_features[i] = features[i]
        # return normalized_features


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        np_features = np.array(features)
        col_max = np.max(np_features, axis=0)
        col_min = np.min(np_features, axis=0)

        for feature in features:
            # print(feature)
            for i in range(len(feature)):
                # print(feature[i])
                if col_min[i] != col_max[i]:
                    feature[i] = (feature[i] - col_min[i]) / (col_max[i] - col_min[i])
                else:
                    feature[i] = 0
        return features

