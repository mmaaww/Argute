import spacy
import random
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer
import logging

class SpacyEntityRecognizer:

    def __init__(self):
        self.nlp = spacy.load('en')
        self.it_num = 20

    def getEntities(self, text):
        entities = []
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append((ent.text, ent.label_, ent.start, ent.end))
        return entities

    def trainExistingEntities(self, textlist, entitieslist):
        if(len(textlist) != len(entitieslist)):
            return -1

        train_data = zip(textlist, entitieslist)
        logging.debug(train_data)

        # train_data = numpy.empty((len(textlist), 2))
        # print train_data
        # for idx in range(0, len(textlist)):
        #    train_data[idx, 0] = textlist[idx]
        #    train_data[idx, 1] = entitieslist[idx]
        # print train_data

        std_entity_types = self.nlp.entity.cfg[u'actions']
        # print nlp.entity.cfg[u'extra_labels']

        entityLabels = list()
        for entities in entitieslist:
            for entity in entities:
                if(not entityLabels.__contains__(entity[2])):
                    entityLabels.append(entity[2])
        for entity_label in entityLabels:
            if(not entity_label in std_entity_types[u'1']):
                self.nlp.entity.add_label(entity_label)
                print "adding label " + entity_label

        print self.nlp.entity.cfg[u'actions']
        print self.nlp.entity.cfg[u'extra_labels']

        # add word to vocab
        for raw_text, _ in train_data:
            doc = self.nlp.make_doc(unicode(raw_text))
            for word in doc:
                _ = self.nlp.vocab[word.orth]

        # train model
        for int in range(self.it_num):
            random.shuffle(train_data)
            for raw_text, entities in train_data:
                doc = self.nlp.make_doc(unicode(raw_text))
                gold = GoldParse(doc, entities=entities)
                self.nlp.tagger(doc) # is it needed for entity model training?
                loss = self.nlp.entity.update(doc, gold)
                self.nlp.end_training()

    def saveModel(self, output_dir):
        self.nlp.save_to_directory(output_dir)
