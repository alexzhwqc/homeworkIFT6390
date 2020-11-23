import pickle
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=0.03,
                 batch_size=100,
                 seed=0,
                 activation="relu",
                 init_method="glorot",
                 normalization=False
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
            if normalization:
                self.normalize()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        # **************************************************************************
        # self.weights is a dictionary with keys w1, b1, w2, b2, ..., wm, bm where m - 1 is the number of hidden layers
        # w1: number_feature   X  hidden_layer_1
        # b0:               1  X  hidden_layer_1
        # w1:  hidden_layer_1  X  hidden-layer_2 X
        # b1:               1  X  hidden_layer_2
        # w3:  hidden_layer_2  X  n_classes
        # b3:               1  X  n_classes
        # **************************************************************************
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            d = np.sqrt(6 / (all_dims[layer_n - 1] + all_dims[layer_n]))
            self.weights[f"W{layer_n}"] = np.random.uniform(low=-d, high=d, size=(all_dims[layer_n-1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]), dtype=float)

    def relu(self, x, grad=False):
        if grad == False:
            return np.maximum(x, 0)
        else:
            # grad = 1 if x>0; 0 otherwise
            return 1 * (x > 0)

    def sigmoid(self, x, grad=False):
        # sigmoid(x) = 1.0/(1.0+np.exp(-x))
        sigm = 1.0/(1.0+np.exp(-x))
        if grad == False:
            # sigmoid(x) = 1.0/(1.0+np.exp(-x))
            return sigm
        else:
            # grad of sigmoid(x) = sigmoid(x)*(1-sigmoid(x))
            return sigm * (1 - sigm)


    def tanh(self, x, grad=False):
        tan = (np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x))
        if grad == False:
            # tanh(x) = (np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x))
            return tan
        else:
            # derivative of tanh(x) = 1-(tanh(x))**2
            return 1 - np.square(tan)

    def leakyrelu(self, x, grad=False):
        alpha = 0.01
        if grad == False:
            # leakyrelu(x) = x if x>0; x*alpha otherwise
            return np.where(x > 0, x, x * alpha)
        else:
            # derivative of leakyrelu(x) = 1.0 if x>0; alpha otherwise
            d_leakyrelu = np.ones_like(x, dtype=float)
            d_leakyrelu[x<=0] = alpha
            return d_leakyrelu

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)

        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)

        elif self.activation_str == "tanh":
            # WRITE CODE HERE
            return self.tanh(x, grad)

        elif self.activation_str == "leakyrelu":
            return self.leakyrelu(x, grad)

        else:
            raise Exception("invalid")
            return 0

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)
        # *********************************************************************
        #z = x
        #z_norm = np.exp(z - np.max(z, axis=1, keepdims=True))
        #return (np.divide(z_norm, np.sum(z_norm, axis=1, keepdims=True)))
        # *********************************************************************
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # softmax(x) = exp(x_i) / (exp(x_0)+...+exp(x_k))
        # ************************************************************************
        # e_x = np.exp(x-np.max(x))
        # return e_x / e_x.sum(axis=-1, keepdims=True)

        # ************************************************************************
        # specially, assuming that x is a matrix of mini_batch X n_classes
        # set axis=-1 to obtain sum of each row of [ e_x / e_x.sum(...)] = 1,
        # set axit= 0 ot obtain sum of each column of [ e_x / e_x.sum(...)] = 1

    def forward(self, x):
        cache = {"Z0": x}

        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE

        for layer_n in range(1, self.n_hidden + 1):
            cache[f"A{layer_n}"] = np.dot(cache[f"Z{layer_n - 1}"], self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"], grad=False)

        layer_n = layer_n + 1
        cache[f"A{layer_n}"] = np.dot(cache[f"Z{layer_n - 1}"], self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
        cache[f"Z{layer_n}"] = self.softmax(cache[f"A{layer_n}"])

        return cache

        # w1: neural_number_hidden_0 X number_feature              b0: neural_number_hidden_0
        # w2: neural_number_hidden_1 X neural_number_hidden_0      b1: neural_number_hidden_1
        # w3: number_classes X neural_number_hidden_1              b2: number_classes
        # ************************ step 1 *******************************
        # Z0: number_feature                        Z0=x
        # A1: minibatch X neural_number_hidden_1    A0=np.dot(Z0, w1)+b1
        # Z1: size_Z1=size_A1                       Z1=self.activation(A1)
        # ************************ step 2 *******************************
        # A2: minibatch X neural_number_hidden_2    A1=np.dot(Z1, w2)+b2
        # Z2: size_Z2=size_A2                       Z2=self.activation(A2)
        # ************************ step 3 ********************************
        # A3: minibatch X number_classes            A2=np.dot(Z2, w3)+b3
        # Z3: size_Z3=size_A3                       Z3=self.softmax(A3)

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}

        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE

        layer_n = self.n_hidden+1 # layer=3
        grads[f"dA{layer_n}"] = output - labels
        len_labels = len(labels)

        dw3 = np.dot(cache[f"Z{layer_n - 1}"].T, grads[f"dA{layer_n}"]) / len_labels
        db3 = np.sum(grads[f"dA{layer_n}"], axis=0, keepdims=True) / len_labels
        # dw3 = np.dot(cache[f"Z{layer_n - 1}"].T, grads[f"dA{layer_n}"])
        # db3 = np.sum(grads[f"dA{layer_n}"], axis=0, keepdims=True)
        dZ2 = np.dot(grads[f"dA{layer_n}"], self.weights[f"W{layer_n}"].T)

        grads[f"dW{layer_n}"] = dw3
        grads[f"db{layer_n}"] = db3
        grads[f"dZ{layer_n - 1}"] = dZ2

        for layer in range(self.n_hidden, 0, -1):
            grads[f"dA{layer}"] = self.activation(cache[f"A{layer}"], grad=True) * grads[f"dZ{layer}"]
            dw = np.dot(cache[f"Z{layer-1}"].T, grads[f"dA{layer}"]) / len_labels
            db = np.sum(grads[f"dA{layer}"], axis=0, keepdims=True) / len_labels
            dZ = np.dot(grads[f"dA{layer}"], self.weights[f"W{layer}"].T)

            grads[f"dW{layer}"] = dw
            grads[f"db{layer}"] = db
            grads[f"dZ{layer - 1}"] = dZ

            #dw = np.dot(cache[f"Z{layer - 1}"].T, grads[f"A{layer}"])
            #db = np.sum(grads[f"A{layer}"], axis=0, keepdims=True)
            # ***************************************************************************
            # w1: number_feature X hidden_layer_1
            # b1:              1 X hidden_layer_1
            # w2: hidden-layer_1 X hidden_layer_2
            # b2:              1 X hidden_layer_2
            # w3: hidden_layer_2 X n_classes
            # b3:              1 X n_classes
            # ***************************************************************************
            # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
            # WRITE CODE HERE

            # dA3: minibatch X n_classes
            # calculate dA3 = maxsoft(X)-onehot(y) = cache-labels
            # dw3 : neural_number_hidden_2 X n_classes
            # db3 :                      1 X n_classes
            # ************************ step 1 ***********************************
            # dA2: minibatch X neural_number_hidden_2    A2=self.activation(Z3)
            # dZ2: minibatch X neural_number_hidden_1    Z2=np.dot(A2, w2.T)
            # ************************ step 0 ***********************************
            # A1: minibatch X neural_number_hidden_1    A0=np.dot(x, w0.T)+b0
            # Z1: size_Z1=size_A0                       Z1=self.activation(A0)

        pass
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden+2):
            # WRITE CODE HERE
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - self.lr * grads[f"db{layer}"]
            pass

    def one_hot(self, y):
        return np.squeeze(np.eye(self.n_classes)[y.reshape(-1)])


    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        # WRITE CODE HERE
        # loss : a scalar
        mean_loss = np.mean(np.sum(-np.log(prediction)*labels, axis=1)) # here calculate the loss of each (x, y)
        # loss = np.sum(-np.log(prediction) * labels) # here calculate the total loss of one minibatch.
        return mean_loss


    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden+1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden+1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)
        print('begin train')
        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))
        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)
                pass

            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test

        # WRITE CODE HERE
        loss = []
        accuracy = []
        y_onehot = self.one_hot(y_test)
        n_batches = int(np.ceil(X_test.shape[0] / self.batch_size))
        for batch in range(n_batches):
            test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
            loss.append(test_loss)
            accuracy.append(test_accuracy)

        mean_loss = np.mean(loss)
        mean_accuracy = np.mean(accuracy)

        return mean_loss, mean_accuracy

    def normalize(self):
        # WRITE CODE HERE
        # compute mean and std along the first axis
        X_train, y_train = self.train
        norm_X_train = (X_train-np.mean(X_train, axis=0))/np.std(X_train, axis=0)
        X_valid, y_valid = self.valid
        norm_X_valid = (X_valid - np.mean(X_valid, axis=0)) / np.std(X_valid, axis=0)
        X_test, y_test = self.test
        norm_X_test = (X_valid - np.mean(X_valid, axis=0)) / np.std(X_valid, axis=0)
        pass
        return norm_X_train, norm_X_valid, norm_X_test

if __name__ == '__main__':
    nn = NN(datapath="svhn.pkl")
    nn.train_loop(30)
    print(nn.train_logs['train_accuracy'])
    print(nn.train_logs['validation_accuracy'])