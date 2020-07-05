import numpy as np
import spacy


class Document(object):
    def __init__(self, _id, _url, _text, _default_categories, _paragraph_type, _gcloud_categories, _tokenized):
        self.id = _id
        self.url = _url
        self.text = _text
        self.defaultCategories = _default_categories
        self.paragraphType = _paragraph_type
        self.gcloudCategories = _gcloud_categories
        self.tokenized = _tokenized
