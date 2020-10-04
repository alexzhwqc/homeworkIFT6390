import numpy as np

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

def minkowski_mat(x, Y, p=2):
    return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

class Q1:

    def feature_means(self, banknote):
        # (a) Q1.feature_means : An array containing the empirical means of each feature,
        # from all examples present in the dataset.
        # Make sure to maintain the original order of features.
        # e.g.:  Q1.feature_means(banknote) =  [μ1,μ2,μ3,μ4]
        # banknote n*5

        data = banknote[:, :-1]
        μ = np.mean(data, axis=0)
        return μ

    def covariance_matrix(self, banknote):
        data = banknote[:, :-1]
        covMatrix = np.cov(data, rowvar=False)
        return covMatrix

    def feature_means_class_1(self, banknote):
        ind_class1 = [i for i in range(banknote.shape[0]) if banknote[i, -1] == 1]
        cμ = np.mean(banknote[ind_class1, :-1], axis=0)
        return cμ

    def covariance_matrix_class_1(self, banknote):
        ind_class1 = [i for i in range(banknote.shape[0]) if banknote[i, -1] == 1]
        cCovMatrix = np.cov(banknote[ind_class1, :-1], rowvar=False)
        return cCovMatrix

class HardParzen:
    def __init__(self, h):
        self.h = h  # h is the threshold distance, h is a positive real.
        self.dist_func = minkowski_mat

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(data[:, -1])  # the elements of self.label_list are monotonically increasing
        self.n_classes = len(self.label_list)

    def compute_predictions(self, test_data):
        pred_test_labels = np.zeros((test_data.shape[0]))

        # For each test datapoint
        for (i, ex) in enumerate(test_data):
            # i is the row index
            # ex is the i'th row

            # Find the distances between ex to training_inputs point using dist_func
            distances = minkowski_mat(ex, self.train_inputs, p=2)
            # find the neighbour index with distance less than self.h
            neighbour_index = [j for j in range(len(distances)) if distances[j] < self.h]

            if len(neighbour_index) == 0:
                k = draw_rand_label(ex, self.label_list)  # k = a random class for ex
                pred_test_labels[i] = k
            else:
                neighb_train_labels = self.train_labels[neighbour_index]

                # create an array count_class len(self.label_list) to save the count classes.
                pre_class = self.label_list[0]
                max_count = 0
                for (m, a) in enumerate(self.label_list):
                    # for each ex, count number of the different classes.
                    index_neigh_set = [j for j in range(len(neighb_train_labels)) if neighb_train_labels[j] == a]
                    if len(index_neigh_set) > max_count:
                        max_count = len(index_neigh_set)
                        pre_class = self.label_list[m]

                pred_test_labels[i] = pre_class

        pred_test_labels_int = pred_test_labels.astype(int)
        return pred_test_labels_int

class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma
        self.sigma_sq = self.sigma**2  # sigma_sq is the square of sigma

    def train(self, train_inputs, train_labels):
        self.n_dims = train_inputs.shape[1]
        self.mu = np.zeros(self.n_dims)

        self.train_inputs = train_inputs
        self.train_labels = train_labels

        self.label_list = np.unique(data[:, -1])
        self.n_classes = len(self.label_list)

        # onehot_train_lables is for predicting the class of test
        # the function to_categorical() to code onehot
        self.onehot_train_labels = to_categorical(self.train_labels)

        # Here, calculate the mean and variance of the train_data data and put it in self.mu
        # self.sigma_sq is given
        self.mu = np.mean(self.train_inputs, axis=0)

    def GaussianKernel(self, dist=0):
        # calculate Gaussian Kernel
        # dist is the distance between two points, dim is the dimension number
        part1 = 1 / ((2 * math.pi) ** (self.n_dims / 2) * (self.sigma_sq ** self.n_dims))
        part2 = math.exp(-1 / 2 * (dist ** 2) / self.sigma_sq)
        return part1 * part2

    def compute_predictions(self, test_data):
        classes_pred = np.zeros(test_data.shape[0], dtype=int) - 1  # classes_pred initialize -1
        self.onehot_train_lables = to_categorical(self.train_labels)

        # For each test datapoint
        for (i, ex) in enumerate(test_data):
            # i is the row index
            # ex is the i'th row test_data

            # count train_inputs.shape[0] by n_classes,
            # it is used to record the multiplication between GaussianKernel and onehot-coding
            counts = np.zeros_like(self.onehot_train_lables, dtype=float)  #

            # Find the distances to each training set point using dist_func
            # distances = self.dist_func(ex, self.train_inputs)
            distances = minkowski_mat(ex, self.train_inputs, p=2)

            total_kernel = 0.0
            for (j, dist) in enumerate(distances):
                # Go through the training set to calculate GaussianKernel * onehot_train_labels[i]
                # Implement SoftRBFParzen with hard window parameter self.mu and self.sigma_sq here.
                kernel = self.GaussianKernel(dist)
                total_kernel += kernel
                # the two way to calculate counts
                counts[j] = kernel * self.onehot_train_labels[j]

            sum_counts = np.sum(counts, axis=0) / total_kernel
            classes_pred[i] = self.label_list[np.argmax(
                sum_counts)]  # np.argmax return the index of max element in sum_counts, it is the index of class.

        return classes_pred


def split_dataset(banknote):
    data = banknote
    label_list = np.unique(data[:, -1])
    n_classes = len(np.unique(data[:, -1]))

    # sperate the indexes into three different sets
    train_indexes = [i for i in range(data.shape[0]) if i % 5 == 0 or i % 5 == 1 or i % 5 == 2]
    valid_indexes = [i for i in range(data.shape[0]) if i % 5 == 3]
    test_indexes = [i for i in range(data.shape[0]) if i % 5 == 4]

    # by the indexes of three sets, copy data into their sets
    train_set = data[train_indexes, :]
    valid_set = data[valid_indexes, :]
    test_set = data[test_indexes, :]

    # draw out the labels from train_set, valid_set and test_set.
    train_labels = train_set[:, -1].astype('int32')
    valid_labels = valid_set[:, -1].astype('int32')
    test_labels = test_set[:, -1].astype('int32')

    # draw out the feature from train_set, valid_set and test_set
    train_data = train_set[:, :-1]
    valid_data = valid_set[:, :-1]
    test_data = test_set[:, :-1]

    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels, label_list, n_classes


def confusion_matrix(true_labels, pred_labels):
    matrix = np.zeros((n_classes, n_classes))

    for (true, pred) in zip(true_labels, pred_labels):
        matrix[int(true - 1), int(pred - 1)] += 1

    return matrix


def comput_test_error(conf_mat):
    # compute test_error from n by n comfusion_matrix
    sum_preds = np.sum(conf_mat)
    sum_correct = np.sum(np.diag(conf_mat))

    return 1.0 - float(sum_correct) / float(sum_preds)


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.h = 1.0
        self.sigma_sq = 1.0

    def hard_parzen(self, h):
        self.h = h
        x_hard_parzen = HardParzen(self.h)
        x_hard_parzen.train(self.x_train, self.x_val)
        y_pred_test_labels = x_hard_parzen.compute_predictions(self.y_train)
        y_confusion_matrix = confusion_matrix(self.y_val, y_pred_test_labels)
        y_error_rate = comput_test_error(y_confusion_matrix)

        y_hard_parzen = HardParzen(self.h)
        y_hard_parzen.train(self.y_train, self.y_val)
        x_pred_test_labels = y_hard_parzen.compute_predictions(self.x_train)
        x_confusion_matrix = confusion_matrix(self.x_val, x_pred_test_labels)
        x_error_rate = comput_test_error(x_confusion_matrix)
        return y_error_rate, x_error_rate

    def soft_parzen(self, sigma):
        self.sigma_sq = sigma ** 2
        x_soft_RBFParzen = SoftRBFParzen(self.sigma_sq)
        x_soft_RBFParzen.train(self.x_train, self.x_val)
        y_pred_test_labels = x_soft_RBFParzen.compute_predictions(self.y_train)
        y_confusion_matrix = confusion_matrix(self.y_val, y_pred_test_labels)
        y_error_rate = comput_test_error(y_confusion_matrix)

        y_soft_RBFParzen = SoftRBFParzen(self.sigma_sq)
        y_soft_RBFParzen.train(self.y_train, self.y_val)
        x_pred_test_labels = y_soft_RBFParzen.compute_predictions(self.x_train)
        x_confusion_matrix = confusion_matrix(self.x_val, x_pred_test_labels)
        x_error_rate = comput_test_error(x_confusion_matrix)

        return y_error_rate, x_error_rate


def get_test_errors(banknote):
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels, label_list, n_classes = split_dataset(
        data)
    # the value star_h is the one (among the proposed set in question 5)
    # that results in the smallest validation error for Parzen with hard window
    x_hard_parzen = HardParzen(star_h)
    x_hard_parzen.train(train_data, train_labels)
    y_hp_pred_test_lab = x_hard_parzen.compute_predictions(test_data)
    y_hp_conf_matrix = confusion_matrix(test_labels, y_hp_pred_test_lab)
    y__hp_error_rate = comput_test_error(y_hp_conf_matrix)

    star_sigma_sq = star_sigma ** 2
    # σ∗  is the parameter (among the proposed set in question 5)
    # that results in the smallest validation error for Parzen with RBF.
    x_soft_RBFParzen = SoftRBFParzen(star_sigma_sq)
    x_soft_RBFParzen.train(train_data, train_labels)
    y_soft_pred_test_lab = x_soft_RBFParzen.compute_predictions(test_data)
    y_soft_conf_matrix = confusion_matrix(test_labels, y_soft_pred_test_lab)
    y_soft_error_rate = comput_test_error(y_soft_conf_matrix)

    # expected output is an array of size 2,
    # the first value being the error rate on the test set of Hard Parzen with parameter ℎ*
    # the second value being the error rate on the test set of Soft RBF Parzen with parameter σ∗
    list_error_rate = [y__hp_error_rate, y_soft_error_rate]
    hp_sp_error_rate = np.array(list_error_rate, dtype=float)

    return hp_sp_error_rate


def random_projections(X, A):
    proj_X = np.dot(X, A) / math.sqrt(2)
    return proj_X