import tensorflow as tf
import numpy as np
import requests
import json
import spacy
from requests.exceptions import HTTPError
from modules.corpus import Corpus
from modules.document import Document
from modules.constants import GlobalConstants
from modules.rnn_classifier import RnnClassifier


class WebPageAnalyzer:
    def __init__(self):
        self.classifiersDict = {}
        self.tensorflowSession = tf.Session()
        self.nlp = spacy.load("en_core_web_lg")
        self.tokenizer = self.nlp.Defaults.create_tokenizer(self.nlp)
        self.corpus = Corpus()
        # Load Classifiers
        for classifier_name, params in GlobalConstants.CLASSIFIERS.items():
            classifier = RnnClassifier(corpus=self.corpus, classifier_name=classifier_name)
            classifier.build_classifier()
            classifier.load_trained_classifier(run_id=params[0], target_category=classifier_name, iteration=params[1],
                                               sess=self.tensorflowSession)
            self.classifiersDict[classifier_name] = classifier

    def get_category_confidence(self, posteriors_list):
        all_posteriors = np.concatenate(posteriors_list, axis=0)
        mean_confidences = np.mean(all_posteriors, axis=0)
        return mean_confidences

    def analyze_page(self, page_url):
        try:
            response = requests.get(
                'https://gtest1.dcats.sigmard.com/api/v1/scrape-text',
                params={'url': page_url},
            )
            # If the response was successful, no Exception will be raised
            response.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  # Python 3.6
            return
        except Exception as err:
            print(f'Other error occurred: {err}')  # Python 3.6
            return
        else:
            print('Success!')
        json_response = response.json()
        # Create documents from relevant Json entries
        documents = []
        for json_field in GlobalConstants.SCRAPE_ENTRIES_TO_LOOK:
            for text in json_response["scrape_result"][json_field]:
                # Normalize text
                normalized_text = Corpus.normalize_text(text)
                # spacy_doc = nlp(normalized_text)
                tokenized_text = self.tokenizer(normalized_text)
                doc_object = Document(_id=len(documents), _url=page_url, _text=normalized_text,
                                      _default_categories=None,
                                      _paragraph_type=None,
                                      _gcloud_categories=None,
                                      _tokenized=tokenized_text)
                documents.append(doc_object)
        # Analyze documents for possible categories
        for category, classifier in self.classifiersDict.items():
            document_posteriors = classifier.analyze_documents(sess=self.tensorflowSession, documents=documents,
                                                               batch_size=1000)
            mean_confidences = self.get_category_confidence(posteriors_list=document_posteriors)
            print("Category:{0} Confidence:{1}".format(category, mean_confidences[1]))


if __name__ == "__main__":
    web_page_analyzer = WebPageAnalyzer()
    web_page_analyzer.analyze_page(page_url="https://edition.cnn.com/")
