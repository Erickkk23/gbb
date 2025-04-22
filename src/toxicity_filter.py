import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.metrics import classification_report # type: ignore
from typing import *

class ToxicityFilter:
    """
    Our ToxicityFilters will be tools for detecting vulgar, offensive, and otherwise toxic
    text messages as might appear in web forums with the purposes of aiding in moderation.
    These are implemented in the present endeavor as a Naive Bayes Classifier (NBC).
    
    [!] You are free to choose whatever attributes needed to implement the ToxicityFilter
    BUT MUST include the following:
    
    Attributes:
        self.vectorizer [CountVectorizer]:
            The vectorizer used to preprocess the text of comments into a format
            amenable to conversion into feature CPTs of the NBC 
    """

    def __init__(self, text_train: pd.DataFrame, labels_train: pd.DataFrame):
        """
        Creates a new forum comment ToxicityFilter trained on the given wikitalk 
        messages and their associated toxic/not labels. Performs any necessary
        preprocessing before training the ToxicityFilter's Naive Bayes Classifier.
        As part of this process, trains and stores the CountVectorizer used
        in the feature extraction process.
        
        Parameters:
            text_train (pd.DataFrame):
                The data frame containing the text of comments on which this NBC
                is to be trained.
            
            labels_train (pd.DataFrame):
                The labels of 0 = non-toxic and 1 = toxic corresponding to each
                row (by index) in the text_train data frame's comments.
        """
        # [!] TODO: Implement the constructor tasks here!
        return
        
    def classify (self, text_test: list[str]) -> list[int]:
        """
        Takes as input a list of raw forum comments, uses the filter's
        vectorizer to transform these into the known bag of words, and then
        returns a list of classifications, one for each input text.
        
        [!] Note: Should only use the vectorizer's transform method, you
        should NOT be using fit_transform (which will re-fit it to test data)
        
        Parameters:
            text_test (list[str]):
                A list of forum comments comprising the messages that the toxicity
                filter must classify as toxic (y=1) or not (y=0)
        
        Returns:
            list[int]:
                A list of classifications, one for each input message, where the
                index of the output class corresponds to the index of input message.
                The ints represent the classes such that y=0=non-toxic and y=1=toxic
        """
        # [!] TODO: Replace this with the proper return from classifying all of the input messages!
        return []
    
    def test_model (self, text_test: pd.DataFrame, labels_test: pd.DataFrame) -> tuple[str, dict]:
        """
        Takes the test-set as input (2 DataFrames consisting of test inputs
        and their associated labels), classifies each, and then prints
        the classification_report on the expected vs. given labels.
        
        Parameters:
            text_test [pd.DataFrame]:
                Pandas DataFrame consisting of the test rows of comments
            labels_test [pd.DataFrame]:
                Pandas DataFrame consisting of the test rows of labels pertaining 
                to each text message
        
        Returns:
            tuple[str, dict]:
                Returns the classification report in two formats as a tuple:
                [0] = The classification report as a prettified string table
                [1] = The classification report in dictionary format
                In either format, contains information on the accuracy of the
                classifier on the test data.
        """
        prediction = self.classify(text_test.values.tolist())
        return (classification_report(labels_test,prediction, output_dict = False),
                classification_report(labels_test,prediction, output_dict = True))
    
