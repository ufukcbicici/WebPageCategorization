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
        print("Initializing Corpus")
        self.tensorflowSession = tf.Session()
        self.nlp = spacy.load("en_core_web_lg")
        self.tokenizer = self.nlp.Defaults.create_tokenizer(self.nlp)
        self.corpus = Corpus()
        print("Corpus Initialized")
        # Load Classifiers
        print("Initializing Classifiers")
        for classifier_name, params in GlobalConstants.CLASSIFIERS.items():
            classifier = RnnClassifier(corpus=self.corpus, classifier_name=classifier_name)
            classifier.build_classifier()
            classifier.load_trained_classifier(run_id=params[0], target_category=classifier_name, iteration=params[1],
                                               sess=self.tensorflowSession)
            self.classifiersDict[classifier_name] = classifier
        print("Classifiers Initialized")

    def get_category_confidence(self, posteriors_list):
        all_posteriors = np.concatenate(posteriors_list, axis=0)
        mean_confidences = np.mean(all_posteriors, axis=0)
        return mean_confidences

    def get_document_from_text(self, text, _id, page_url):
        # Normalize text
        normalized_text = Corpus.normalize_text(text)
        # spacy_doc = nlp(normalized_text)
        tokenized_text = self.tokenizer(normalized_text)
        doc_object = Document(_id=_id, _url=page_url, _text=normalized_text,
                              _default_categories=None,
                              _paragraph_type=None,
                              _gcloud_categories=None,
                              _tokenized=tokenized_text)
        return doc_object

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
        documents = {}
        categories_accepted = []
        for json_field in GlobalConstants.SCRAPE_ENTRIES_TO_LOOK:
            documents[json_field] = []
            if json_field in {"alt_items", "heading_items", "link_items", "title_items"}:
                for text in json_response["scrape_result"][json_field]:
                    doc_object = self.get_document_from_text(text=text, _id=-1, page_url=page_url)
                    documents[json_field].append(doc_object)
            elif json_field == "bsoup":
                plain_text = json_response["scrape_result"]["bsoup"]["plain_text"]
                doc_object = self.get_document_from_text(text=plain_text, _id=-1, page_url=page_url)
                documents[json_field].append(doc_object)

        # Analyze documents for possible categories
        for category, classifier in self.classifiersDict.items():
            category_posteriors = []
            for field_name, field_documents in documents.items():
                field_posteriors = classifier.analyze_documents(sess=self.tensorflowSession,
                                                                documents=field_documents,
                                                                batch_size=1000)
                assert len(field_posteriors) == len(field_documents)
                # mean_confidences = self.get_category_confidence(posteriors_list=document_posteriors)
                field_posterior = np.mean(np.concatenate(field_posteriors, axis=0), axis=0)
                print("Category:{0} Field:{1} Count:{2} Confidence:{3}".format(category, field_name,
                                                                               len(field_posteriors),
                                                                               field_posterior[1]))
                category_posteriors.append(np.concatenate(field_posteriors, axis=0))
            category_posteriors = [np.mean(_p, axis=0)[np.newaxis, :] for _p in category_posteriors]
            category_posterior = np.mean(np.concatenate(category_posteriors, axis=0), axis=0)
            print("Category:{0} Confidence:{1}".format(category, category_posterior[1]))
            if category_posterior[1] >= 0.5:
                categories_accepted.append(category)
        print("Recognized Web Page Categories:{0}".format(categories_accepted))


if __name__ == "__main__":
    web_page_analyzer = WebPageAnalyzer()
    web_page_analyzer.analyze_page(page_url="https://edition.cnn.com/")
