import tensorflow as tf
import os
import pathlib
import numpy as np
import pickle
from sklearn.metrics import classification_report

from collections import Counter
# from auxillary.db_logger import DbLogger
# from global_constants import GlobalConstants, DatasetType
# from model.deep_classifier import DeepClassifier
from modules.constants import GlobalConstants
from modules.db_logger import DbLogger
from modules.deep_classifier import DeepClassifier


class RnnClassifier(DeepClassifier):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.initial_state = None
        self.initial_state_fw = None
        self.initial_state_bw = None
        self.finalLstmState = None
        self.outputs = None
        self.attentionMechanismInput = None
        self.contextVector = None
        self.alpha = None
        self.finalState = None
        self.temps = []

    def get_embeddings(self):
        super().get_embeddings()
        # FC Layers
        self.inputs = tf.layers.dense(self.inputs, GlobalConstants.DENSE_INPUT_DIMENSION, activation=tf.nn.relu)
        if GlobalConstants.USE_INPUT_DROPOUT:
            self.inputs = tf.nn.dropout(self.inputs, keep_prob=self.keep_prob)

    @staticmethod
    def get_stacked_lstm_cells(hidden_dimension, num_layers):
        cell_list = [tf.contrib.rnn.LSTMCell(hidden_dimension,
                                             forget_bias=1.0,
                                             state_is_tuple=True) for _ in range(num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)
        return cell

    def get_classifier_structure(self):
        num_layers = GlobalConstants.NUM_OF_LSTM_LAYERS
        if not GlobalConstants.USE_BIDIRECTIONAL_LSTM:
            cell = RnnClassifier.get_stacked_lstm_cells(hidden_dimension=GlobalConstants.LSTM_HIDDEN_LAYER_SIZE,
                                                        num_layers=num_layers)
            # Add dropout to cell output
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            # Dynamic LSTM
            with tf.variable_scope('LSTM'):
                self.outputs, state = tf.nn.dynamic_rnn(cell,
                                                        inputs=self.inputs,
                                                        initial_state=self.initial_state,
                                                        sequence_length=self.sequence_length)

            final_state = state
            self.finalLstmState = final_state[num_layers - 1].h
        else:
            cell_fw = RnnClassifier.get_stacked_lstm_cells(hidden_dimension=GlobalConstants.LSTM_HIDDEN_LAYER_SIZE,
                                                           num_layers=num_layers)
            cell_bw = RnnClassifier.get_stacked_lstm_cells(hidden_dimension=GlobalConstants.LSTM_HIDDEN_LAYER_SIZE,
                                                           num_layers=num_layers)
            # Add dropout to cell output
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)
            # Init states
            self.initial_state_fw = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
            self.initial_state_bw = cell_bw.zero_state(self.batch_size, dtype=tf.float32)
            # Dynamic Bi-LSTM
            with tf.variable_scope('Bi-LSTM'):
                self.outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                      cell_bw,
                                                                      inputs=self.inputs,
                                                                      initial_state_fw=self.initial_state_fw,
                                                                      initial_state_bw=self.initial_state_bw,
                                                                      sequence_length=self.sequence_length)
                final_state_fw = state[0][num_layers - 1]
                final_state_bw = state[1][num_layers - 1]
                self.finalLstmState = tf.concat([final_state_fw.h, final_state_bw.h], 1)
        if GlobalConstants.USE_ATTENTION_MECHANISM:
            self.add_attention_mechanism()
        else:
            self.finalState = self.finalLstmState

    def add_attention_mechanism(self):
        if GlobalConstants.USE_BIDIRECTIONAL_LSTM:
            forward_rnn_outputs = self.outputs[0]
            backward_rnn_outputs = self.outputs[1]
            self.attentionMechanismInput = tf.concat([forward_rnn_outputs, backward_rnn_outputs], axis=2)
        else:
            self.attentionMechanismInput = self.outputs
        with tf.variable_scope('Attention-Model'):
            hidden_state_length = self.attentionMechanismInput.get_shape().as_list()[-1]
            self.contextVector = tf.Variable(name="context_vector",
                                             initial_value=tf.random_normal([hidden_state_length], stddev=0.1))
            w = self.contextVector
            H = self.attentionMechanismInput
            M = tf.tanh(H)
            M = tf.reshape(M, [-1, hidden_state_length])
            w = tf.reshape(w, [-1, 1])
            pre_softmax = tf.reshape(tf.matmul(M, w), [-1, self.max_sequence_length])
            zero_mask = tf.equal(pre_softmax, 0.0)
            replacement_tensor = tf.fill([self.batch_size, self.max_sequence_length], -1e100)
            masked_pre_softmax = tf.where(zero_mask, replacement_tensor, pre_softmax)
            self.alpha = tf.nn.softmax(masked_pre_softmax)
            r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                          tf.reshape(self.alpha, [-1, self.max_sequence_length, 1]))
            r = tf.squeeze(r)
            h_star = tf.tanh(r)
            h_drop = tf.nn.dropout(h_star, self.keep_prob)
            self.finalState = h_drop
            self.temps.append(pre_softmax)
            self.temps.append(zero_mask)
            self.temps.append(masked_pre_softmax)

    def get_softmax_layer(self):
        hidden_layer_size = GlobalConstants.LSTM_HIDDEN_LAYER_SIZE
        num_of_classes = self.corpus.get_num_of_classes()
        # Softmax output layer
        with tf.name_scope('softmax'):
            if not GlobalConstants.USE_BIDIRECTIONAL_LSTM:
                softmax_w = tf.get_variable('softmax_w', shape=[hidden_layer_size, num_of_classes], dtype=tf.float32)
            elif GlobalConstants.USE_BIDIRECTIONAL_LSTM:
                softmax_w = tf.get_variable('softmax_w', shape=[2 * hidden_layer_size, num_of_classes],
                                            dtype=tf.float32)
            else:
                raise NotImplementedError()
            softmax_b = tf.get_variable('softmax_b', shape=[num_of_classes], dtype=tf.float32)
            # self.l2_loss += tf.nn.l2_loss(softmax_w)
            # self.l2_loss += tf.nn.l2_loss(softmax_b)
            self.logits = tf.matmul(self.finalState, softmax_w) + softmax_b
            self.posteriors = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.posteriors, 1, name='posteriors')

    def train(self, **kwargs):
        target_category = kwargs["target_category"]
        run_id = DbLogger.get_run_id()
        explanation = RnnClassifier.get_explanation()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData, col_count=2)
        # self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=None)
        file_path = pathlib.Path(__file__).parent.absolute()
        model_folder = os.path.join(file_path, "..", "models", target_category)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        losses = []
        for iteration in range(GlobalConstants.ITERATION_COUNT):
            sequences_arr, seq_lengths, labels_arr = self.corpus.get_training_batch(target_category=target_category)
            feed_dict = {self.batch_size: sequences_arr.shape[0],
                         self.input_x: sequences_arr,
                         self.input_y: labels_arr,
                         self.keep_prob: GlobalConstants.DROPOUT_KEEP_PROB,
                         self.sequence_length: seq_lengths,
                         self.max_sequence_length: GlobalConstants.MAX_SEQUENCE_LENGTH}
            run_ops = [self.optimizer, self.mainLoss]
            results = self.sess.run(run_ops, feed_dict=feed_dict)
            losses.append(results[1])
            iteration += 1
            if iteration % 10 == 0:
                avg_loss = np.mean(np.array(losses))
                losses = []
                print("Iteration:{0} Avg Loss:{1}".format(iteration, avg_loss))
            if iteration % 100 == 0:
                checkpoint_folder = os.path.join(model_folder, "lstm{0}_iteration{1}".format(run_id, iteration))
                path = os.path.join(checkpoint_folder, "lstm{0}_iteration{1}.ckpt".format(run_id, iteration))
                saver.save(self.sess, path)

    def test(self, **kwargs):
        target_category = kwargs["target_category"]
        batch_size = kwargs["batch_size"]
        data_type = kwargs["data_type"]
        file_path = pathlib.Path(__file__).parent.absolute()
        model_folder = os.path.join(file_path, "..", "models")
        posteriors = []
        ground_truths = []
        doc_id = 0
        for sequences_arr, seq_lengths, labels_arr in \
                self.corpus.get_document_sequences(target_category=target_category, data_type=data_type):
            batch_id = 0
            while batch_id < sequences_arr.shape[0]:
                seq_batch = sequences_arr[batch_id:batch_id + batch_size]
                feed_dict = {self.batch_size: seq_batch.shape[0],
                             self.input_x: seq_batch,
                             self.keep_prob: GlobalConstants.DROPOUT_KEEP_PROB,
                             self.sequence_length: seq_lengths[batch_id:batch_id + batch_size],
                             self.max_sequence_length: GlobalConstants.MAX_SEQUENCE_LENGTH}
                run_ops = [self.posteriors]
                results = self.sess.run(run_ops, feed_dict=feed_dict)
                ground_truths.append(labels_arr[batch_id:batch_id + batch_size])
                posteriors.append(results[0])
                batch_id += batch_size
                print("\rProcessing document:{0}".format(doc_id), end="")
                if len(posteriors) % 1000 == 0:
                    y = np.concatenate(ground_truths)
                    y_hat = np.argmax(np.concatenate(posteriors, axis=0), axis=1)
                    report = classification_report(y_true=y, y_pred=y_hat, target_names=["Other", target_category])
                    print(report)
                    model_file = open(os.path.join(model_folder, "{0}_ground_truths.sav".format(data_type)), "wb")
                    pickle.dump(ground_truths, model_file)
                    model_file.close()
                    model_file = open(os.path.join(model_folder, "{0}_posteriors.sav".format(data_type)), "wb")
                    pickle.dump(posteriors, model_file)
                    model_file.close()
