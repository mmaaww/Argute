# deprecated file

from argute.engine.util.parse.spaCy.TextClassifier import SpaCyTextClassifier

classifier = SpaCyTextClassifier()

# wrapper method to train the classifier
def trainClassifier(dataTrain, labelsTrain):
    classifier.train(dataTrain,labelsTrain)

# wrapper method to validate/test the classifier
def validateClassifier(dataTest, labelsTest):
    classifier.validate(dataTest, labelsTest)

def getTestClassification(stmt):
    return classifier.predictSingle(stmt)

def print10MostFeaturesOfClassifier():
    classifier.print10MostFeaturesToUse()

# method to get the parsing result
def getParsingResult():
    return ""

