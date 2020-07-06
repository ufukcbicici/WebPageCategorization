from enum import Enum


class DatasetTypes(Enum):
    training = 0
    validation = 1
    test = 2


class GlobalConstants:
    EMBEDDING_SIZE = 300
    BATCH_SIZE = 64
    L2_LAMBDA_COEFFICENT = 0.0
    DROPOUT_KEEP_PROB = 1.0
    # RNN Structure
    MAX_SEQUENCE_LENGTH = 100
    SLIDING_WINDOW_SIZE = 10
    DENSE_INPUT_DIMENSION = 256
    USE_INPUT_DROPOUT = False
    NUM_OF_LSTM_LAYERS = 1
    USE_BIDIRECTIONAL_LSTM = True
    LSTM_HIDDEN_LAYER_SIZE = 256
    USE_ATTENTION_MECHANISM = True
    ITERATION_COUNT = 100000
    # Web Service
    SCRAPE_ENTRIES_TO_LOOK = ["alt_items"]
    CLASSIFIERS = {"adult": (4, 30000), "news": (5, 50000)}

    def __init__(self):
        pass
