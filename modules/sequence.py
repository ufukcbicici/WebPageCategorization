import numpy as np


class Sequence:
    def __init__(self, document_id, label, tokens_list, is_training=1):
        self.documentId = document_id
        self.isTraining = is_training
        self.label = label
        self.tokenArr = np.array(tokens_list, dtype=np.int32)

    # def get_db_row(self):
    #     sequence_str = ""
    #     for i in range(self.tokenArr.shape[0]):
    #         sequence_str += "{0}".format(np.asscalar(self.tokenArr[i]))
    #         if i < self.tokenArr.shape[0] - 1:
    #             sequence_str += ","
    #     row = (int(self.label), self.documentId, self.isTraining, sequence_str)
    #     return row
    #
    # @staticmethod
    # def get_sequence_from_db_row(row):
    #     assert len(row) == 4
    #     label = row[0]
    #     document_id = row[1]
    #     is_training = row[2]
    #     tokenstr_list = row[3].split(",")
    #     token_ids = [int(token_str) for token_str in tokenstr_list]
    #     return Sequence(document_id=document_id, label=label, tokens_list=token_ids, is_training=is_training)