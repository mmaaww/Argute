# this is test function. After test, all the functions will be move into pipeline to form a more prehensive product-grade NLP processing

from argute.engine.util.parse.spaCy.TextClassifier import SpaCyTextClassifier
from argute.engine.util.parse.spaCy.EntityRecognizer import SpacyEntityRecognizer
from argute.engine.util.parse.coreNLP.CoreNLPManager import CoreNLPManager
from argute.engine.core.pipeline.Pipeline_MultiStage import Pipeline_MultiStage
from argute.engine.util import UtilManager
import numpy as np
import logging
import datetime

tc_train_data_file = UtilManager.tc_train_data_file
tc_test_data_file = UtilManager.tc_test_data_file
ner_train_data_file = UtilManager.ner_train_data_file
ner_test_data_file = UtilManager.ner_test_data_file
model_output_dir = UtilManager.model_output_dir

spaCyTextClassifier = SpaCyTextClassifier()
spaCyEntityRecognizer = SpacyEntityRecognizer()
coreNLPManager = CoreNLPManager()

pipelineManager = Pipeline_MultiStage()

def cleanData(dataInput):
    # clean the whilespace in the front and at the tail of each element
    cleaner = lambda str: str.rstrip(' ').strip(' ')
    vfunc = np.vectorize(cleaner)
    dataInput = vfunc(dataInput)
    logging.debug(dataInput)
    return dataInput

def loadData(filename):
    # load data
    dtype1 = np.dtype([('sample', np.str), ('label', np.str)])
    dataInput = np.loadtxt(filename, dtype=np.str, delimiter='|')

    # clean the data
    dataInput = cleanData(dataInput)
    return dataInput

def loadAndTrainTextClassifier():
    # load data
    dataTrainInput = loadData(tc_train_data_file)

    # separate the data from labels
    dataTrain = dataTrainInput[:,0]
    labelsTrain = dataTrainInput[:,1]

    spaCyTextClassifier.train(dataTrain, labelsTrain)

def loadAndTrainEntityRecognizer():
    # load data
    dataTrainInput = np.loadtxt(ner_train_data_file, dtype=np.str, delimiter='|')

    # separate the data from entities
    dataTrain = dataTrainInput[:, 0]
    entitiesTrainText = dataTrainInput[:, 1]

    # convert to unicode, the code doesn't work, it can not transform to the unicode
    for idx in range(0, len(dataTrain)):
        dataTrain[idx] = dataTrain[idx].rstrip(' ').strip(' ')
    u_dataTrain = unicode(dataTrain)
    # print u_dataTrain

    # separate entity
    entitiesTrain = list()
    for entitiesText in entitiesTrainText:
        entitiesList = list()
        entityListText = entitiesText.split('/')
        for entity in entityListText:
            entriesText = entity.split(',')
            entitiesList.append((int(entriesText[0]), int(entriesText[1]), unicode(entriesText[2].rstrip(' ').strip(' '))))
        entitiesTrain.append(entitiesList)
    # print dataTrain
    # print entitiesTrain

    spaCyEntityRecognizer.trainExistingEntities(dataTrain, entitiesTrain)

def validateTextClassifier():
    dataTestInput = np.loadtxt(tc_test_data_file, dtype=np.str, delimiter='|')
    dataTestInput = cleanData(dataTestInput)

    dataTest = dataTestInput[:, 0]
    labelsTest = dataTestInput[:, 1]

    spaCyTextClassifier.validate(dataTest, labelsTest)

def print10MostFeaturesOfClassifier():
    spaCyTextClassifier.print10MostFeaturesToUse()

def interactiveMode():
    while(True):
        text = raw_input(">>")
        if text == 'quit':
            break
        print "result from spaCy: \n" + \
            "classification: \t " + str(spaCyTextClassifier.predictSingle(text)) + "\n" + \
            "entity: \t " + str(spaCyEntityRecognizer.getEntities(unicode(text))) + "\n" + \
            "result from CoreNLP: \n" + \
            "entity: \t" + str(coreNLPManager.getNE(text)) + "\n"

def main():
    # setup logger
    logging.basicConfig(filename=UtilManager.log_file, level=logging.DEBUG)
    logging.debug("===== starting ====")

    # main code text classfier
    loadAndTrainTextClassifier()
    # validateTextClassifier()
    # print10MostFeaturesOfClassifier()
    print "testing trained spaCy Text Classifier "
    print spaCyTextClassifier.predictSingle("I feel like listen to some romantic music")
    print "testing trained CoreNLP Text Classifier (coming soon ...) "

    # main code for entity recognizer
    print "testing Spacy Entity Recognizer (original) "
    print spaCyEntityRecognizer.getEntities(u"I want to go to San Francisco International Airport")
    print spaCyEntityRecognizer.getEntities(u"I like to play Gone with the Wind")
    print "testing CoreNLP Entity Recognizer (Original) "
    print coreNLPManager.getNE("I want to go to San Francisco International Airport")
    print coreNLPManager.getNE("I like to play Gone with the Wind")

    print "testing Spacy Entity Recognizer (retrained) "
    loadAndTrainEntityRecognizer()
    print spaCyEntityRecognizer.getEntities(u"I want to go to San Francisco International Airport")
    print spaCyEntityRecognizer.getEntities(u"I like to play Gone with the Wind")
    print spaCyEntityRecognizer.getEntities(u"Please play Gone with the Wind")
    print spaCyEntityRecognizer.getEntities(u"Please play Gone with Wind")
    print spaCyEntityRecognizer.getEntities(u"I like to play Gangnam Style")    # the fail of this test case indicates that training data needs to do coverage with list of songs

    print "testing Multiple Stage pipeline (coming soon ...) \n"

    interactiveMode()

if __name__ == "__main__":
    main()
