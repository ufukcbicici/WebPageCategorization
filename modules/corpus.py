import json
import logging
import os
import numpy as np
import pathlib
import regex
from os import listdir
from os.path import isfile, join
from collections import Counter

from modules.constants import GlobalConstants
from modules.document import Document
import gensim.downloader as api
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Corpus:
    FILTER_REGEX = "[^a-zA-Z\d\s]|[\r\n]"
    EMAIL_MATCHER = "[a-zA-Z0-9\_\.\+\-]++@[a-zA-Z0-9\-]++\.[a-zA-Z0-9\-\.]++"
    URL_MATCHER = "((?<=\s)pic\.|www\.|https?\:\s?)[^\s]++|[0-9a-z\.]+\.([Cc](?:[Oo](?:[Mm]))|N(?:ET|et)|TR|tr|e(?:du|u)|E(?:DU|U)|net|org|GOV|gov|ORG|fr|FR|az|DE|RU|de|ru)[^\'\"\s]*+"
    DATE_MATCHER = "(?<=[\s\n])[0-9]{2}+([\.\-\/])[0-9]{2}+\1[0-9]{4}+(?!\1)"
    NUMBER_MATCHER = "(?:(?<=\s)\-)?(?<![0-9\.]|(?<!\sNO)(?<!\sK)\:)[0-9]++(([\,][0-9]++)++|(\.[0-9]++)++(\,[0-9]++)?|(?=[^\.\,\:])|[\.\,]++(?=$|\s))"

    # WORD_TOKENIZER_REGEX = "([^a-zıçöşüğ0-9\s]+)?(\s+([^a-zıçöşüğ0-9\s]+|https?\:[^\s]+|www\.[^\s]+)?)+"
    WORD_TOKENIZER_REGEX = "(?:(?:[^A-ZİÇÖŞÜĞa-zıçöşüğ0-9\s]++)?(?:\s++(?:[^A-ZİÇÖŞÜĞa-zıçöşüğ0-9\s]++)?)++|(?:[^A-ZİÇÖŞÜĞa-zıçöşüğ\s]++(?=$)))"
    # tokens = regex.split("([^a-zıçöşüğ0-9\s]+)?(\s+([^a-zıçöşüğ0-9\s]+|https?\:[^\s]+|www\.[^\s]+)?)+", sentence)
    EMAIL_TOKEN = "EMAIL"
    URL_TOKEN = "URL"
    DATE_TOKEN = "DATE"
    NUMBER_TOKEN = "NUMBER"

    def __init__(self, n_grams=1):
        self.documents = None
        self.trainIds = None
        self.testIds = None
        self.embeddings = Corpus.load_word_embeddings()
        self.tfIdfVectorizer = TfidfVectorizer(stop_words="english",
                                               norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,
                                               ngram_range=(1, n_grams))
        self.positiveNegativeDocsDict = {}

    def create_tf_idf_analyzer(self):
        # Create key word corpus
        keyword_lists = [document.keyWordsPart for document in self.documents]
        self.tfIdfVectorizer.fit(raw_documents=keyword_lists)

    @staticmethod
    def save_word_embeddings():
        file_path = pathlib.Path(__file__).parent.absolute()
        model_folder = os.path.join(file_path, "..", "models")
        model = api.load("word2vec-google-news-300")
        model_file = open(os.path.join(model_folder, "google_embeddings.sav"), "wb")
        pickle.dump(model, model_file)
        model_file.close()

    @staticmethod
    def load_word_embeddings():
        file_path = pathlib.Path(__file__).parent.absolute()
        model_folder = os.path.join(file_path, "..", "models")
        model_file = open(os.path.join(model_folder, "google_embeddings.sav"), "rb")
        model = pickle.load(model_file)
        model_file.close()
        return model

    @staticmethod
    def get_num_of_classes():
        return 2

    @staticmethod
    def normalize_text(text):
        normalized_text = text
        normalized_text = normalized_text.replace("\n", " ")
        # Step 1: Find and Replace Emails
        normalized_text = regex.sub(Corpus.EMAIL_MATCHER, Corpus.EMAIL_TOKEN, normalized_text)
        # Step 2: Find and Replace URLs
        normalized_text = regex.sub(Corpus.URL_MATCHER, Corpus.URL_TOKEN, normalized_text)
        # Step 3: Find and Replace Dates
        normalized_text = regex.sub(Corpus.DATE_MATCHER, Corpus.DATE_TOKEN, normalized_text)
        # Step 4:
        normalized_text = regex.sub(Corpus.NUMBER_MATCHER, Corpus.NUMBER_TOKEN, normalized_text)
        # Step 5:
        normalized_text = regex.sub(Corpus.FILTER_REGEX, "", normalized_text)
        return normalized_text

    def read_documents(self, load_from_hd=False):
        # Read Json files
        file_path = pathlib.Path(__file__).parent.absolute()
        model_folder = os.path.join(file_path, "..", "models")
        data_folder = os.path.join(file_path, "..", "documents")
        if load_from_hd is False:
            nlp = spacy.load("en_core_web_lg")
            tokenizer = nlp.Defaults.create_tokenizer(nlp)
            self.documents = []
            files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
            json_values = []
            for file_path in files:
                json_file = open(os.path.join(data_folder, file_path))
                json_content = json_file.read()
                json_list = json.loads(json_content)["samples"]
                assert isinstance(json_list, list)
                json_values.extend(json_list)
                json_file.close()
            # Create document objects
            document_set = set()
            for doc_id, json_value in enumerate(json_values):
                print("Processing doc: {0}/{1}".format(doc_id + 1, len(json_values)))
                doc_key = (json_value["text"], json_value["default_categories"], json_value["paragraph_type"])
                if doc_key in document_set:
                    continue
                document_set.add(doc_key)
                # Normalize text
                normalized_text = Corpus.normalize_text(json_value["text"])
                # spacy_doc = nlp(normalized_text)
                tokenized_text = tokenizer(normalized_text)
                doc_object = Document(_id=json_value["id"], _url=json_value["url"], _text=normalized_text,
                                      _default_categories=json_value["default_categories"],
                                      _paragraph_type=json_value["paragraph_type"],
                                      _gcloud_categories=json_value["gcloud_categories"],
                                      _tokenized=tokenized_text)
                self.documents.append(doc_object)
            # Keep statistics
            label_counter = Counter([doc_obj.defaultCategories for doc_obj in self.documents])
            print("label_counter={0}".format(label_counter))
            model_file = open(os.path.join(model_folder, "processed_documents.sav"), "wb")
            pickle.dump(self.documents, model_file)
            model_file.close()
        else:
            model_file = open(os.path.join(model_folder, "processed_documents.sav"), "rb")
            self.documents = pickle.load(model_file)
            model_file.close()
        self.documents = np.array(self.documents)

    def prepare_train_test_sets(self, test_ratio=0.1, load_from_hd=False):
        file_path = pathlib.Path(__file__).parent.absolute()
        model_folder = os.path.join(file_path, "..", "models")
        if load_from_hd is False:
            documents_index_array = np.arange(len(self.documents))
            self.trainIds, self.testIds = train_test_split(documents_index_array, test_size=test_ratio)
            for set_type, ids in zip(["training_docs_ids", "test_docs_ids"], [self.trainIds, self.testIds]):
                model_file = open(os.path.join(model_folder, "{0}.sav".format(set_type)), "wb")
                pickle.dump(ids, model_file)
                model_file.close()
        else:
            ids_loaded = []
            for set_type in ["training_docs_ids", "test_docs_ids"]:
                model_file = open(os.path.join(model_folder, "{0}.sav".format(set_type)), "rb")
                ids = pickle.load(model_file)
                model_file.close()
                ids_loaded.append(ids)
            self.trainIds = ids_loaded[0]
            self.testIds = ids_loaded[1]

    def get_sequence_from_document(self, document, sequence_start_index, target_category):
        word_embeddings = []
        max_sequence_length = GlobalConstants.MAX_SEQUENCE_LENGTH
        for idx in range(max_sequence_length):
            if sequence_start_index + idx >= len(document.tokenized):
                break
            word_string = document.tokenized[sequence_start_index + idx].text
            if word_string not in self.embeddings.wv:
                word_embeddings.append(np.zeros(shape=(self.embeddings.wv.vector_size,)))
            else:
                word_embeddings.append(self.embeddings.wv[word_string])
        word_embeddings = np.stack(word_embeddings, axis=0)
        seq_length = word_embeddings.shape[0]
        label = int(document.defaultCategories == target_category)
        return word_embeddings, seq_length, label

    def get_training_batch(self, target_category, positive_negative_ratio=0.5, document_sample_count=32,
                           sequences_per_document=2):
        # Sample from documents
        max_sequence_length = GlobalConstants.MAX_SEQUENCE_LENGTH
        batch_size = document_sample_count * sequences_per_document
        positive_doc_samples_count = int(document_sample_count * positive_negative_ratio)
        negative_doc_samples_count = document_sample_count - positive_doc_samples_count
        if target_category not in self.positiveNegativeDocsDict:
            positive_doc_ids = [idx for idx in self.trainIds
                                if self.documents[idx].defaultCategories == target_category]
            negative_doc_ids = [idx for idx in self.trainIds
                                if self.documents[idx].defaultCategories != target_category]
            assert len(positive_doc_ids) > 0
            self.positiveNegativeDocsDict[target_category] = (positive_doc_ids, negative_doc_ids)
        else:
            positive_doc_ids, negative_doc_ids = self.positiveNegativeDocsDict[target_category]
        positive_doc_samples = np.random.choice(positive_doc_ids, positive_doc_samples_count, replace=False)
        negative_doc_samples = np.random.choice(negative_doc_ids, negative_doc_samples_count, replace=False)
        positive_doc_samples = self.documents[positive_doc_samples]
        negative_doc_samples = self.documents[negative_doc_samples]
        positive_sequence_samples = [np.random.choice(len(doc.tokenized), sequences_per_document)
                                     for doc in positive_doc_samples]
        negative_sequence_samples = [np.random.choice(len(doc.tokenized), sequences_per_document)
                                     for doc in negative_doc_samples]
        sequences_arr = np.zeros(shape=(batch_size, max_sequence_length, self.embeddings.wv.vector_size))
        labels_arr = np.zeros(shape=(batch_size,), dtype=np.int32)
        seq_lengths = np.zeros(shape=(batch_size,), dtype=np.int32)
        # Prepare actual sequences
        index_in_batch = 0
        for sequence_ids_list, doc_samples in zip([negative_sequence_samples, positive_sequence_samples],
                                                  [negative_doc_samples, positive_doc_samples]):
            for sequence_ids, doc_sample in zip(sequence_ids_list, doc_samples):
                for sequence_id in sequence_ids:
                    word_embeddings, seq_length, label = \
                        self.get_sequence_from_document(document=doc_sample, sequence_start_index=sequence_id,
                                                        target_category=target_category)
                    seq_lengths[index_in_batch] = seq_length
                    sequences_arr[index_in_batch, 0:seq_length, :] = word_embeddings
                    labels_arr[index_in_batch] = int(doc_sample.defaultCategories == target_category)
                    index_in_batch += 1
        assert index_in_batch == batch_size
        return sequences_arr, seq_lengths, labels_arr

    def get_document_sequences(self, target_category, data_type, outside_documents=None):
        max_sequence_length = GlobalConstants.MAX_SEQUENCE_LENGTH
        window_size = GlobalConstants.SLIDING_WINDOW_SIZE
        if outside_documents is None:
            if data_type == "train":
                docs = self.documents[self.trainIds]
            else:
                docs = self.documents[self.testIds]
        else:
            docs = outside_documents
        for doc_id, document in enumerate(docs):
            seq_id = 0
            embeddings = []
            seq_lengths = []
            labels = []
            while seq_id < len(document.tokenized):
                word_embeddings, seq_length, label = \
                    self.get_sequence_from_document(document=document, sequence_start_index=seq_id,
                                                    target_category=target_category)
                embeddings.append(word_embeddings)
                seq_lengths.append(seq_length)
                labels.append(label)
                seq_id += window_size
            sequences_arr = np.zeros(shape=(len(embeddings), max_sequence_length, self.embeddings.wv.vector_size))
            for idx in range(len(embeddings)):
                assert embeddings[idx].shape[0] == seq_lengths[idx]
                sequences_arr[idx, 0:seq_lengths[idx], :] = embeddings[idx]
            seq_lengths_arr = np.array(seq_lengths)
            labels_arr = np.array(labels)
            yield sequences_arr, seq_lengths_arr, labels_arr


if __name__ == "__main__":
    # Mock tokens
    # Corpus.save_word_embeddings()
    corpus = Corpus()
    corpus.read_documents(load_from_hd=True)
    corpus.prepare_train_test_sets(test_ratio=0.1, load_from_hd=True)
    corpus.get_training_batch(target_category="adult")
