# setup spaCy
from spacy.en import English
parser = English()

# import spaCy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re
import logging

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", """, """, "'ve"]

# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")

    # replace twitter @mentions
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mentionFinder.sub("@MENTION", text)

    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")

    # lowercase
    text = text.lower()
    return text

# Every step in a pipeline needs to be a "transformer".
# Define a custom transformer to clean text using spaCy
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """
    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens


class SpaCyTextClassifier:

    # setup global variables for pipe
    # the vectorizer and classifer to use
    # note that I changed the tokenizer in CountVectorizer to use a custom function using spaCy's tokenizer
    vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1, 1))
    clf = LinearSVC()
    # the pipeline to clean, tokenize, vectorize, and classify
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])

    def __int__(self):
        return True

    def train(self, dataTrain, labelsTrain, dataTest = None, labelsTest = None):
        # train
        self.pipe.fit(dataTrain, labelsTrain)

        # validate
        if (dataTest != None):
            self.validate(dataTest, labelsTest)

    def validate(self, data, labels):
        # test, need more work on the debugger 
        preds = self.pipe.predict(data)

        # Todo: solve the following problem, toString function doesn't work
        # logging.debug("----------------------------------------------------------------------------------------------")
        # logging.debug("results:")
        # for (sample, pred) in zip(data, preds):
        #    logging.debug(sample, ":" , pred)
        # logging.debug("accuracy:" , accuracy_score(labels, preds))

    def predictSingle(self, stmt):
        data = [stmt]
        preds = self.pipe.predict(data)
        return preds[0]

    def printNMostInformative(self, N):
        """Prints features with the highest coefficient values, per class"""
        feature_names = self.vectorizer.get_feature_names()
        coefs_with_fns = sorted(zip(self.clf.coef_[0], feature_names))
        topClass1 = coefs_with_fns[:N]
        topClass2 = coefs_with_fns[:-(N + 1):-1]
        logging.debug("Class 1 best: ")
        for feat in topClass1:
            logging.debug(feat)
        logging.debug("Class 2 best: ")
        for feat in topClass2:
            logging.debug(feat)

    def print10MostFeaturesToUse(self):
        logging.debug("----------------------------------------------------------------------------------------------")
        logging.debug("Top 10 features used to predict: ")
        # show the top features
        self.printNMostInformative(10)