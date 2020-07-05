import numpy as np
import os
import pathlib
from modules.corpus import Corpus
from modules.db_logger import DbLogger
from modules.rnn_classifier import RnnClassifier


def train_model(target_category, load_from_hd):
    file_path = pathlib.Path(__file__).parent.absolute()
    DbLogger.log_db_path = os.path.join(file_path, "..", "db.db")
    corpus = Corpus()
    corpus.read_documents(load_from_hd=load_from_hd)
    corpus.prepare_train_test_sets(test_ratio=0.1, load_from_hd=load_from_hd)
    classifier = RnnClassifier(corpus=corpus)
    classifier.build_classifier()
    classifier.train(target_category=target_category)
    print("X")


def run_model(run_id, target_category, iteration):
    file_path = pathlib.Path(__file__).parent.absolute()
    DbLogger.log_db_path = os.path.join(file_path, "..", "db.db")
    corpus = Corpus()
    corpus.read_documents(load_from_hd=True)
    corpus.prepare_train_test_sets(test_ratio=0.1, load_from_hd=True)
    classifier = RnnClassifier(corpus=corpus)
    classifier.build_classifier()
    classifier.load_trained_classifier(run_id=run_id, target_category=target_category, iteration=iteration)
    classifier.test(batch_size=1000, target_category="adult", data_type="train")
    print("X")


if __name__ == "__main__":
    # train_model(target_category="adult", load_from_hd=True)
    run_model(run_id=4, target_category="adult", iteration=30000)
    # import time
    #
    # print('Start.')
    # for i in range(100):
    #     time.sleep(0.02)
    #     print("\rProcessing document:{0}".format(i), end="")
    #     # print('\rDownloading File FooFile.txt [%d%%]' % i, end="")
    # print('\nDone.')
