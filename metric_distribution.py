import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
import json
import math
from textwrap import dedent
from collections import namedtuple
from inspect import getdoc
from typing import List, Dict, Iterable, Callable

### Metric Functions

def metric_total_length(text) -> float:
    '''Total length'''
    return float(len(text))

def metric_avg_sentence_length(text) -> float:
    '''Average sentence length'''
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(text.strip())
    return sum(map(len, sents)) / len(sents)

SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()

def metric_polarity(text) -> float:
    '''Absolute intensity of the polarity of the text'''
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(text.strip())
    scores = SENTIMENT_ANALYZER.polarity_scores(text)
    return scores['compound'] - scores['neg']
