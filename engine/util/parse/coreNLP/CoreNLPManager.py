from argute.engine.util import UtilManager
from stanfordcorenlp import StanfordCoreNLP
import logging

class CoreNLPManager:

    def __init__(self, lang='en', memory='8g', outputFormat='json'):
        self.nlp = StanfordCoreNLP(UtilManager.coreNLP_java_lib, lang=lang, memory=memory)

    def getPOS(self, text):
        return self.nlp.pos_tag(text)

    def getNE(self, text):
        return self.nlp.ner(text)

    def getDep(self, text):
        return self.nlp.dependency_parse(text)

    def getTokens(self, text):
        return self.nlp.word_tokenize(text)
