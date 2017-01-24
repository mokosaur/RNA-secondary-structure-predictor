from BasePredictor import BasePredictor
import mxnet as mx

# import hebel
# from hebel.models import NeuralNet
# from hebel.optimizers import SGD
# from hebel.parameter_updaters import MomentumUpdate
# from hebel.data_providers import BatchDataProvider
# from hebel.monitors import ProgressMonitor
# from hebel.schedulers import exponential_scheduler, linear_scheduler_up

import numpy as np
import rna

import theano
import theano.tensor as T

import lasagne


class NaivePredictor(BasePredictor):
    def __init__(self, sequence_length, substrings=False, max_examples=500, library='mxnet'):
        super().__init__()
        self.sequence_length = sequence_length
        self.max_examples = max_examples
        self.substrings = substrings
        self.library = library

    def preprocess(self):
        X = self.X
        y = []
        list = []
        for i in X:
            if self.substrings:
                m = rna.Molecule(i[0, 0], i[0, 1])
                for j in m.get_substrings(self.sequence_length):
                    seq = j.seq
                    dot = j.dot
                    if dot[::-1] in y:
                        seq = seq[::-1]
                        dot = dot[::-1]
                    list.append(rna.encode_rna(seq))
                    y.append(dot)
                    # list.append(rna.encode_rna(j.seq))
                    # y.append(j.dot)
            else:
                if len(i[0, 0]) == self.sequence_length:
                    seq = i[0, 0]
                    dot = i[0, 1]
                    if dot[::-1] in y:
                        seq = seq[::-1]
                        dot = dot[::-1]
                    list.append(rna.encode_rna(seq))
                    y.append(dot)
        X = np.array(list)
        y = y[:self.max_examples]
        z = set(y)
        self.num_labels = len(z)
        self.a = {}
        idx = 0
        for i in z:
            self.a[i] = idx
            idx += 1
        for i in range(len(y)):
            y[i] = self.a[y[i]]
        y = np.array(y)
        return X[:self.max_examples, :], y[:self.max_examples]

    def train_X(self):
        X, y = self.preprocess()

        if self.library == 'mxnet':
            data = mx.sym.Variable('data')

            fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=self.num_labels * 10)
            act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

            # The second fully-connected layer and the according activation function
            fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=self.num_labels * 5)
            act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

            # The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
            fc4 = mx.sym.FullyConnected(data=act2, name='fc4', num_hidden=self.num_labels)
            # The softmax and loss layer
            mlp = mx.sym.SoftmaxOutput(data=fc4, name='softmax')
            # create a model
            # mx.viz.plot_network(symbol=mlp, shape={"data": (28, 22)}).render("NaiveNet", view=True)
            examples = mx.io.NDArrayIter(X, y)

            import logging
            logging.basicConfig(level=logging.INFO)
            self.model = mx.model.FeedForward(symbol=mlp,
                                              num_epoch=350,
                                              learning_rate=0.001,
                                              wd=0.00001,
                                              momentum=0.9)

            self.model.fit(X=examples)
        # if self.library == 'hebel':
        #     # Create model object
        #     train_data = BatchDataProvider(X, y)
        #
        #     model = NeuralNet(n_in=train_data.D, n_out=self.num_labels,
        #                       layers=[2000, 2000, 2000, 500],
        #                       activation_function='relu',
        #                       dropout=True, input_dropout=0.2)
        #
        #     # Create optimizer object
        #     progress_monitor = ProgressMonitor(
        #         experiment_name='rna',
        #         save_model_path='examples/rna',
        #         save_interval=5,
        #         output_to_log=True)
        #
        #     optimizer = SGD(model, MomentumUpdate, train_data, train_data, progress_monitor,
        #                     learning_rate_schedule=exponential_scheduler(5., .995),
        #                     momentum_schedule=linear_scheduler_up(.1, .9, 100))
        #
        #     # Run model
        #     optimizer.run(50)
        if self.library == 'lasagne':
            input_var = T.matrix('inputs')
            target_var = T.ivector('targets')

            l_in = lasagne.layers.InputLayer(shape=(None, self.sequence_length),
                                             input_var=input_var)
            l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
            l_hid1 = lasagne.layers.DenseLayer(
                l_in_drop, num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
            l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

            l_hid2 = lasagne.layers.DenseLayer(
                l_hid1_drop, num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify)

            l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
            l_out = lasagne.layers.DenseLayer(
                l_hid2_drop, num_units=self.num_labels,
                nonlinearity=lasagne.nonlinearities.softmax)

            prediction = lasagne.layers.get_output(l_out)
            loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
            params = lasagne.layers.get_all_params(l_out, trainable=True)
            updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)

            f_learn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
            self.model = theano.function([input_var], prediction, allow_input_downcast=True)

            # Training
            it = 3500
            for i in range(it):
                l = f_learn(X, y)

    def predict(self, seq):
        prob = [[], []]
        dot = ''
        if self.library == 'mxnet':
            example = mx.io.NDArrayIter(np.array([rna.encode_rna(seq), rna.encode_rna(seq[::-1])]))
            prob = self.model.predict(example)

        if self.library == 'lasagne':
            prob = self.model(np.array([rna.encode_rna(seq), rna.encode_rna(seq[::-1])]))
        print(prob[0].max(), prob[1].max())
        if prob[0].max() > prob[1].max():
            max = prob[0].argmax()
        else:
            max = prob[1].argmax()
        for i, j in self.a.items():
            if j == max:
                dot = i
                break
        m = rna.Molecule(seq, dot)
        m.show()
        return m
