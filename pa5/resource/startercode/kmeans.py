import numpy as np


#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n)  # this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
    # according to some distribution, first generate a random number between 0 and
    # 1 using generator.rand(), then find the the smallest index n so that the
    # cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    centers = [p]

    for i in range(n_cluster - 1):
        current_centers = x[centers]
        distances = []
        sum = 0
        for point in x:
            distance = np.min(np.sum(np.square(point - current_centers), axis=1))
            sum += distance
            distances.append(distance)

        r = generator.rand()
        curr = 0
        max_distance_idx = 0
        for j in range(len(distances)):
            curr += distances[j] / sum
            if curr > r:
                max_distance_idx = j
                break

        # max_distance_idx = distances.index(max_distance)
        centers.append(max_distance_idx)

    # centers_point = [x[p]]
    # centers = [p]
    #
    #
    # for k in range(n_cluster - 1):
    #     distances = []
    #     for i in range(len(x)):
    #         min_distance = float("inf")
    #         for j in range(len(centers_point)):
    #             center = centers_point[j]
    #             distance = np.sum((x[i] - center) ** 2)
    #             if distance < min_distance:
    #                 min_distance = distance
    #         distances.append(min_distance)
    #     sum_distances = sum(distances)
    #     for m in range(len(distances)):
    #         distances[m] = distances[m] / sum_distances
    #
    #     r = generator.rand()
    #     curr = 0
    #     max_distance = 0
    #     for i in range(len(distances)):
    #         curr += distances[i]
    #         if curr > r:
    #             max_distance = i
    #             break
    #     # index = get_center(generator, distance_to_closest_centroid)
    #     centers.append(max_distance)
    #     centers_point.append(x[max_distance])

            # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


def get_center(generator, distance_to_closest_centroid):
    a = generator.rand()
    cumulative_prob = 0
    for index in range(len(distance_to_closest_centroid)):
        cumulative_prob += distance_to_closest_centroid[index]
        if cumulative_prob > a:
            return index

# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array,
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0),
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        pre_j = -1
        update = 0
        assignments = []
        uk = None
        current_centers = x[self.centers]

        for i in range(self.max_iter):
            update += 1
            assignment = []
            for point in x:
                distance = np.sum(np.square(point - current_centers), axis=1)
                k = np.argmin(distance)
                assignment.append(k)
            assignments = np.array(assignment)

            csums = []
            for c in range(self.n_cluster):
                csum = np.sum(np.square(x[assignments == c] - current_centers[c]))
                csums.append(csum)

            cur_j = np.sum(csums) / N

            if pre_j >= 0:
                if abs(cur_j - pre_j) <= self.e:
                    break

            pre_j = cur_j

            uk = np.zeros((self.n_cluster, D))
            for c in range(self.n_cluster):
                if x[assignments == c].any():
                    uk[c] = np.sum(x[assignments == c], axis=0) / len(x[assignments == c])
                else:
                    uk[c] = current_centers[c]
            current_centers = uk

        return uk, assignments, update


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented,
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        kmeans = KMeans(n_cluster=self.n_cluster, generator=self.generator)
        uk, assignments, update = kmeans.fit(x, centroid_func=centroid_func)

        centroid_labels = []
        for i in range(len(uk)):
            members_idx = np.where(assignments == i)
            members_label = y[members_idx]
            majority = np.argmax(np.bincount(members_label))
            centroid_labels.append(majority)

        centroid_labels = np.array(centroid_labels)
        centroids = uk

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        labels = []
        for point in x:
            distance = np.sum(np.square(point - self.centroids), axis=1)
            k = np.argmin(distance)
            labels.append(self.centroid_labels[k])

        return np.array(labels)


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    d1 = image.shape[0]
    d2 = image.shape[1]

    image_2d = image.reshape(-1, 3)
    replacement = []
    for point in image_2d:
        distance = np.sum(np.square(point - code_vectors), axis=1)
        idx = np.argmin(distance)
        replace = code_vectors[idx]
        replacement.append(replace)

    replacement = np.array(replacement).reshape((d1, d2, 3))

    return replacement
