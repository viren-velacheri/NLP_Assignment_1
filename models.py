# models.py

from sentiment_data import *
from utils import *
import numpy as np
from collections import Counter
import random
import math
# import matplotlib.pyplot as plt
# from sentiment_classifier import *
# import string
class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        sentence = [word.lower() for word in sentence]
        wordsAndCounts = Counter(sentence)
        if add_to_indexer:
            for word in list(wordsAndCounts):
                self.indexer.add_and_get_index(word, True)
        
        return wordsAndCounts
        
        


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        sentence = [word.lower() for word in sentence]
        bigram = []
        for i in range(len(sentence) - 1):
            wordPair = (sentence[i], sentence[i + 1])
            bigram.append(wordPair)
            if add_to_indexer:
                self.indexer.add_and_get_index(wordPair, True)
        
        return Counter(bigram)

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.totalWordCountings = Counter()

    def get_indexer(self):
        return self.indexer
    
    def get_total(self):
        return self.totalWordCountings
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        sentence = [word.lower() for word in sentence]
        wordsAndCounts = Counter(sentence)
        # stopWords = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]
        if add_to_indexer:
            for word in list(wordsAndCounts):
                self.indexer.add_and_get_index(word, True)
        
        for word in list(wordsAndCounts):
            # if word in stopWords:
            #     del wordsAndCounts[word]
            wordsAndCounts[word] = 1
            
        
        # if add_to_indexer:
        #     self.totalWordCountings += wordsAndCounts
        
        return wordsAndCounts
        


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weightVector, featurizer):
        self.weightVector = weightVector
        self.featurizer = featurizer
    def predict(self, sentence: List[str]) -> int:
        indexMapping = self.featurizer.get_indexer()
        featureVector = np.zeros(self.featurizer.get_indexer().__len__())
        wordsAndFrequencies = self.featurizer.extract_features(sentence, False)
        for word in wordsAndFrequencies:
            if indexMapping.index_of(word) != -1:
                featureVector[indexMapping.index_of(word)] = wordsAndFrequencies[word]
            
        predicted = np.sign(np.dot(self.weightVector, featureVector))
        if predicted == -1:
            return 0
        else:
            return predicted


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weightVector, featurizer):
        self.weightVector = weightVector
        self.featurizer = featurizer
    def predict(self, sentence: List[str]) -> int:
        indexMapping = self.featurizer.get_indexer()
        featureVector = np.zeros(self.featurizer.get_indexer().__len__())
        wordsAndFrequencies = self.featurizer.extract_features(sentence, False)
        for word in wordsAndFrequencies:
            if indexMapping.index_of(word) != -1:
                featureVector[indexMapping.index_of(word)] = wordsAndFrequencies[word]
            
        predicted = np.dot(self.weightVector, featureVector)
        predicted = 1.0 / (1.0 + math.exp(-1 * predicted))
        if predicted >= 0.5:
            return 1
        else:
            return 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:

    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    indexMapping = feat_extractor.get_indexer()
    weights = np.zeros(feat_extractor.get_indexer().__len__())
    learning_rate = 0.5
    epochs = 15
    # topHighest = Beam(10)
    # topLowest = Beam(10)
    for t in range(epochs):
        random.shuffle(train_exs) 
        for train_instance in train_exs:
            featureVector = np.zeros(feat_extractor.get_indexer().__len__())
            actual = train_instance.label
            wordsAndFrequencies = feat_extractor.extract_features(train_instance.words, False)
            for word in wordsAndFrequencies:
                if indexMapping.index_of(word) != -1:
                    wordIndex = indexMapping.index_of(word)
                    featureVector[wordIndex] = wordsAndFrequencies[word]
            
            predicted = np.sign(np.dot(weights, featureVector))
            if predicted == -1:
                predicted = 0
            
            
            weights = weights + learning_rate * (actual - predicted) * featureVector
        # learning_rate -= 0.02
    
    # for wordIndex in range(indexMapping.__len__()):
    #     wordScore = weights[wordIndex]
    #     word = indexMapping.get_object(wordIndex)
    #     topHighest.add(word, wordScore)
    #     topLowest.add(word, -1 * wordScore)
    
    # print("HIGHEST WORDS \n")
    # print(topHighest.get_elts())
    # print("LOWEST WORDS \n")
    # print(topLowest.get_elts())
    
    return PerceptronClassifier(weights, feat_extractor)



def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    indexMapping = feat_extractor.get_indexer()
    weights = np.zeros(feat_extractor.get_indexer().__len__())
    learning_rate = 0.5
    epochs = 15
    # listLikelihoods = []
    # listLosses = []
    # xlabels = [x + 1 for x in range(epochs)]
    # accuracies = []
    for t in range(epochs):
        random.shuffle(train_exs)  
        # logLikelihood = 0
        # loss = 0
        # accuracy = evaluate(LogisticRegressionClassifier(weights, feat_extractor), dev_exs)
        # accuracies.append(accuracy)
        for train_instance in train_exs:
            featureVector = np.zeros(feat_extractor.get_indexer().__len__())
            actual = train_instance.label
            wordsAndFrequencies = feat_extractor.extract_features(train_instance.words, False)
            indices = []
            for word in wordsAndFrequencies:
                if indexMapping.index_of(word) != -1:
                    wordIndex = indexMapping.index_of(word)
                    indices.append(wordIndex)
                    featureVector[wordIndex] = wordsAndFrequencies[word]
            
            predicted = np.dot(weights, featureVector)
            predicted = 1.0 / (1.0 + math.exp(-1 * predicted))
            weights = weights + learning_rate * (actual - predicted) * predicted * (1 - predicted) * featureVector
            # logLikelihood += (actual * math.log(predicted) + (1 - actual) * math.log(1 - predicted))
            # loss += -1 * (actual * math.log(predicted) + (1 - actual) * math.log(1 - predicted))
        # logLikelihood /= len(train_exs)
        # loss /= len(train_exs)
        # listLikelihoods.append(logLikelihood)
        # listLosses.append(loss)
    # fig, axs = plt.subplots(3)
    # fig.tight_layout()
    # axs[0].plot(xlabels, listLikelihoods)
    # axs[0].set_title("Log Likelihood")
    # axs[1].plot(xlabels, accuracies)
    # axs[1].set_title("Development Accuracy")
    # axs[2].plot(xlabels, listLosses)
    # axs[2].set_title("Loss (Negative Log Likelihood)")
    # plt.show()

    return LogisticRegressionClassifier(weights, feat_extractor)

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
        for sentExample in train_exs:
            sentence = sentExample.words
            feat_extractor.extract_features(sentence, True)
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
        for sentExample in train_exs:
            sentence = sentExample.words
            feat_extractor.extract_features(sentence, True)
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
        for sentExample in train_exs:
            sentence = sentExample.words
            feat_extractor.extract_features(sentence, True)
        
        # mappings = feat_extractor.get_total()
        # feat_indexer = feat_extractor.get_indexer()
        # for word in list(mappings):
        #     if mappings[word] == 1:
        #         del mappings[word]
        #     else:
        #         feat_indexer.add_and_get_index(word, True)

    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model