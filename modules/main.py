import tensorflow as tf
import numpy as np
import os
import pathlib
from modules.corpus import Corpus
from modules.db_logger import DbLogger
from modules.rnn_classifier import RnnClassifier
import argparse

from modules.web_page_analyzer import WebPageAnalyzer


def train_model(target_category, load_from_hd):
    session = tf.Session()
    file_path = pathlib.Path(__file__).parent.absolute()
    DbLogger.log_db_path = os.path.join(file_path, "..", "db.db")
    corpus = Corpus()
    corpus.read_documents(load_from_hd=load_from_hd)
    corpus.prepare_train_test_sets(test_ratio=0.1, load_from_hd=load_from_hd)
    classifier = RnnClassifier(corpus=corpus, classifier_name=target_category)
    classifier.build_classifier()
    classifier.train(target_category=target_category, session=session)
    print("X")


def run_model(run_id, target_category, iteration):
    session = tf.Session()
    file_path = pathlib.Path(__file__).parent.absolute()
    DbLogger.log_db_path = os.path.join(file_path, "..", "db.db")
    corpus = Corpus()
    corpus.read_documents(load_from_hd=True)
    corpus.prepare_train_test_sets(test_ratio=0.1, load_from_hd=True)
    classifier = RnnClassifier(corpus=corpus, classifier_name=target_category)
    classifier.build_classifier()
    classifier.load_trained_classifier(run_id=run_id, target_category=target_category,
                                       iteration=iteration, sess=session)
    classifier.test(batch_size=1000, target_category=target_category, data_type="test", session=session)
    print("X")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    url = args.url
    web_page_analyzer = WebPageAnalyzer()
    web_page_analyzer.analyze_page(page_url=url)

