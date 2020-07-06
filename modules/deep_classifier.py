import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import os
import pathlib
import numpy as np

from collections import Counter

from modules.constants import GlobalConstants


class DeepClassifier:
    def __init__(self, corpus, classifier_name):
        self.classifierName = classifier_name
        self.corpus = corpus
        self.embeddings = None
        self.inputs = None
        self.logits = None
        self.posteriors = None
        self.predictions = None
        self.numOfCorrectPredictions = None
        self.accuracy = None
        self.optimizer = None
        self.globalStep = None
        self.correctPredictions = None
        self.batch_size = None
        self.input_word_codes = None
        self.input_x = None
        self.input_y = None
        self.keep_prob = None
        self.sequence_length = None
        self.max_sequence_length = None
        self.isTrainingFlag = None
        # L2 loss
        self.mainLoss = None
        self.l2_loss = None

    def build_classifier(self):
        with tf.variable_scope(self.classifierName):
            self.get_inputs()
            self.get_embeddings()
            self.get_classifier_structure()
            self.get_softmax_layer()
            self.get_loss()
            self.get_accuracy()
            self.get_optimizer()

    def get_inputs(self):
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_word_codes = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_word_codes')
        self.input_x = tf.placeholder(dtype=tf.float32,
                                      shape=[None, GlobalConstants.MAX_SEQUENCE_LENGTH,
                                             self.corpus.embeddings.wv.vector_size], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')
        self.max_sequence_length = tf.placeholder(dtype=tf.int32, name='max_sequence_length')
        self.isTrainingFlag = tf.placeholder(name="is_training", dtype=tf.bool)
        self.l2_loss = tf.constant(0.0)

    def get_embeddings(self):
        # vocabulary_size = self.corpus.get_vocabulary_size()
        # embedding_size = GlobalConstants.EMBEDDING_SIZE
        # max_sequence_length = GlobalConstants.MAX_SEQUENCE_LENGTH
        with tf.name_scope('embedding'):
            # self.embeddings = tf.get_variable('embedding',
            #                                   shape=[self.wordEmbeddings.shape[0], self.wordEmbeddings.shape[1]],
            #                                   dtype=tf.float32, trainable=False)
            # self.inputs = tf.nn.embedding_lookup(self.embeddings, self.input_word_codes)
            self.inputs = tf.identity(self.input_x)

    def get_classifier_structure(self):
        pass

    def get_softmax_layer(self):
        pass

    def get_loss(self):
        # Loss
        with tf.name_scope('loss'):
            tvars = tf.trainable_variables()
            # L2 regularization for LSTM weights
            for tv in tvars:
                if 'kernel' in tv.name or "_w" in tv.name:
                    self.l2_loss += tf.nn.l2_loss(tv)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits)
            self.mainLoss = tf.reduce_mean(losses) + GlobalConstants.L2_LAMBDA_COEFFICENT * self.l2_loss

    def get_accuracy(self):
        with tf.name_scope('accuracy'):
            self.correctPredictions = tf.equal(self.predictions, self.input_y)
            self.numOfCorrectPredictions = tf.reduce_sum(tf.cast(self.correctPredictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correctPredictions, tf.float32), name='accuracy')

    def get_optimizer(self):
        # Train procedure
        self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        # Learning rate decay
        # starter_learning_rate = GlobalConstants.INITIAL_LR_CLASSIFIER
        # learning_rate = tf.train.exponential_decay(starter_learning_rate,
        #                                            self.globalStep,
        #                                            GlobalConstants.DECAY_PERIOD_LSTM,
        #                                            GlobalConstants.DECAY_RATE_LSTM,
        #                                            staircase=True)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.mainLoss,
                                                           global_step=self.globalStep)

    def train(self, **kwargs):
        pass

    def test(self, **kwargs):
        pass

    def analyze_documents(self, sess, documents, batch_size):
        pass

    def load_trained_classifier(self, sess, run_id, target_category, iteration):
        tvars = tf.trainable_variables(scope=self.classifierName)
        file_path = pathlib.Path(__file__).parent.absolute()
        model_folder = os.path.join(file_path, "..", "models", target_category)
        checkpoint_folder = os.path.join(model_folder, "lstm{0}_iteration{1}".format(run_id, iteration))
        model_path = os.path.join(checkpoint_folder, "lstm{0}_iteration{1}.ckpt".format(run_id, iteration))
        saved_vars = checkpoint_utils.list_variables(checkpoint_dir=model_path)
        for var in tvars:
            # assert len([_var for _var in saved_vars if _var.name == var.name]) == 1
            # if "Adam" in var.name:
            #     continue
            var_name = var.name[len(self.classifierName)+1:]
            source_array = checkpoint_utils.load_variable(checkpoint_dir=model_path, name=var_name)
            tf.assign(var, source_array).eval(session=sess)

    @staticmethod
    def get_explanation():
        vars_dict = vars(GlobalConstants)
        explanation = ""
        for k, v in vars_dict.items():
            explanation += "{0}: {1}\n".format(k, v)
        return explanation
