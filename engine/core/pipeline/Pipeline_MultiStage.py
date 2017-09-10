# This is the multiple stage NLP process pipeline which aims to enhance the quaility of NLP processing
# It contains the following 5 steps
# 1. Use langauge model to correct any error or clean the data (may leverage n-best from ASR)
# 2. Use highly reliable offline model (CoreNLP) for feature extraction. (NE, POS are 1st priority; dep and coreference later)
# 3. Use Dictionary and/or knowledge Graph to refine and fine-grain the entity recog
# 4. Use online model (customized) to predict the final answer (or train the customized model)
# 5. Convert all the entity into standford format, maybe with additional information, in response
# 6. (Optional) use high quality data to refine the offline model

from argute.engine.core.pipeline import DataLoader
from argute.engine.util import UtilManager
from argute.engine.util.parse.spaCy.EntityRecognizer import SpacyEntityRecognizer
from argute.engine.util.parse.spaCy.TextClassifier import SpaCyTextClassifier
from argute.engine.util.parse.coreNLP.CoreNLPManager import CoreNLPManager
from argute.engine.core.pipeline.Pipe import *
from argute.engine.core.pipeline import Dictionary
import logging

class Pipeline_MultiStage:

    spaCyEntityRecognizer = SpacyEntityRecognizer()
    spaCyTextClassifier = SpaCyTextClassifier()
    coreNLPManager = CoreNLPManager()

    def __init__(self):
        self.name = "mutiple stage pipeline"

    def loadandTrain(self, filename):

        # return [n*3] array, the first column is text, second column is intent, the 3rd column is entity
        dataInput = DataLoader.loadData(filename)
        self.train(dataInput)

    # input format should be like [('text', 'class', [('token', 'label', 'offset_b', 'offset_e')])]
    def train(self, datainput):
        if(datainput == None or (not datainput is list)):
            raise ValueError("the input should be list, instead " + type(datainput))

        # Step 1: for each input
        dataTrain = list()
        classesTrain = list()
        labelsTrain = list()
        for data in datainput:
            pipe = self.process(data)
            data_to_train = pipe.getDataToTrain()
            dataTrain.append(data_to_train[0])
            classesTrain.append(data_to_train[1])
            labelsTrain.append(data_to_train[2])

        # Step 2: train with spaCy (both TC and NER)
        self.spaCyTextClassifier.train(dataTrain, classesTrain)
        self.spaCyEntityRecognizer.trainExistingEntities(dataTrain, labelsTrain)

    # input format should be like ('text', 'class', [('token', 'label', 'offset_b', 'offset_e')])
    def process(self, input):
        if(input == None or (not input is list) or (not len(input) == 3) or (not input[0] is str) or (not input[1] is str) or (not input[2] is list)):
            raise ValueError("the input should be [str, str, list], instead " + type(input))

        # Step 1: convert input into Pipe.pipe structure
        text = input[0]
        clazz = input[1]
        labels = input[2]

        my_pipe = pipe(text)
        my_pipe.setLabels(clazz, labels)

        # Step 2&3: get all aligned entities: self.getAllEntities()
        # the return should be [('token', 'entity', 'offset_b', 'offset_e')]
        entities = self.getAllEntities(text)

        # Step 4: If label not recognizeable by any, add to the local dictionary
        labelTokenList = labels[: 0]
        entityTokenList = entities[: 0]
        for idx in range(0, len(labelTokenList) - 1):
            labelToken = labelTokenList[idx]
            if(not labelToken in entityTokenList):
                Dictionary.insert(labelToken, labels[idx : 1])
                entities.append(labels[idx])

        # Step 5: set entities [('token', 'entity', 'offset_b', 'offset_e')]
        for entity in entities:
            my_pipe.addEntity(entity[0], entity[1], entity[2], entity[3])

        # Step 6: return Pipe.pipe
        return my_pipe

    # TODO: return Pipe.response
    def predict(self, text):
        # Step 1: convert input into Pipe.pipe structure

        # Step 2&3: get all aligned entities: self.getAllEntities()

        # Step 4: set recognized entity

        # Step 5: process NE and TC (spaCy) with data_to_predict

        # Step 6: convert it back to origianl text and offsets

        # Step 7: return Pipe.response

        return True

    # the return should be [('token', 'entity', 'offset_b', 'offset_e')] the offset is on word level, all using unicode
    def getAllEntities(self, text):

        # Step 2a: call CoreNLP NER to get all the entities
        # the CoreNLP return is a list of [(u'I', u'U')] for every word
        coreNLP_entities = list()
        coreNLP_entities__ = self.coreNLPManager.getNE(text)
        # process result into standard format
        token = ''
        entity_type = None
        offset = -1
        for idx in range(0, len(coreNLP_entities__) - 1):
            entity = coreNLP_entities__[idx]
            if(entity[1] == u'0'):
                if(not token == None):
                    coreNLP_entities.append((token, entity_type, offset, idx))
                    token = ''
                    entity_type = None
            else:
                if(entity[1] == entity_type):
                    token.join(entity[0])
                else:
                    entity_type = entity[1]
                    offset = idx
        if(not entity_type == None):
            coreNLP_entities.append((token, entity_type, offset, len(coreNLP_entities__)))

        # TODO: Step 2b: call global dictionary / KG to get all the matching entities
        # input parameter list, output [('token', 'entity', 'offset_b', 'offset_e')] the offset is on word level, all using unicode
        globalDict_entities = list()

        # NOT USE Step 2C and 2D
        # Step 2c: call spaCy local model to get all the entities
        # the spaCy return is [u'I', u'U', int, int], word level offset, only for recognized entity. It is the same format as our format
        #spaCy_entities = self.spaCyEntityRecognizer.getEntities(text)

        # TODO: Step 2d: call local dictionary to get all the matching entities
        # input parameter list, output [('token', 'entity', 'offset_b', 'offset_e')] the offset is on word level, all using unicode
        #localDict_entities = list()

        # Step 3: align entities (no overlapping). weight global ML > global dict > local ML > local dict
        result = coreNLP_entities
        result.append(self.getAddEntities(result, globalDict_entities))
        # result.append(self.getAddEntities(result, spaCy_entities))
        # result.append(self.getAddEntities(result, localDict_entities))

        return result

    def getAddEntities(self, high_priority, low_priority):
        addEntities = list()
        for low_entry in low_priority:
            overlap_mark = -1
            for high_entry in high_priority:
                if((low_entry[2] <= high_entry[2] and low_entry[3] > high_entry[2]) or (low_entry[2] < high_entry[3] and low_entry[3] >= high_entry[3])):
                    # overlaps
                    overlap_mark = 1
                    break
            if(overlap_mark == -1):
                addEntities.append(low_entry)
        return addEntities;

    def loadandTest(self, filename):
        return True

pipelineManager = Pipeline_MultiStage()
def interactiveMode():
    while(True):
        text = raw_input(">>")
        if text == 'quit':
            break
        print "result from multiple stage pipeline: \n" + \
            "classification: \t " + str(pipelineManager.predict(text))

def main():
    # setup logger
    logging.basicConfig(filename=UtilManager.log_file, level=logging.DEBUG)
    logging.debug("===== starting multiple stage pipeline ====")

    print "testing multiple stage pipeline"
    print "loadding the data and train"
    pipelineManager.loadandTrain(UtilManager.tc_train_data_file)

    print "test the model"
    pipelineManager.predict("I want to go to San Francisco International Airport")
    pipelineManager.predict("I like to play Gone with the Wind")

    interactiveMode()

if __name__ == "__main__":
    main()