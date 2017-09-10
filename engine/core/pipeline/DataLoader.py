import logging
import numpy as np

def cleanData(dataInput):
    # clean the whilespace in the front and at the tail of each element
    cleaner = lambda str: str.rstrip(' ').strip(' ')
    vfunc = np.vectorize(cleaner)
    dataInput = vfunc(dataInput)
    logging.debug(dataInput)
    return dataInput

def loadData(filename):
    # load data
    # dtype1 = np.dtype([('sample', np.str), ('label', np.str)])
    dataInput = np.loadtxt(filename, dtype=np.str, delimiter='|')

    # clean the data
    dataInput = cleanData(dataInput)

    # separate text, classes and labels
    textList = dataInput[: 0]
    classesTextList = dataInput[: 1]
    labelsTextList = dataInput[: 2]

    # seperate classes
    classList = list()
    for classesText in classesTextList:
        classList = classesText.split('/')
        classList = cleanData(classList)

    # separate labels
    labelList = list()
    for idx in range(0, len(labelsTextList) - 1):
        labelsText = labelsTextList[idx]
        labelTextList = labelsText.split('/')
        labelTextList = cleanData(labelTextList)
        labelEntryList = list()
        for labelTextSingle in labelTextList:
            labelItems = labelTextSingle.split(',')
            labelItems = cleanData(labelItems)
            label = labelItems[0].rstrip(' ').strip(' ')
            offset_b = int(labelItems[1])
            offset_e = int(labelItems[2])
            token_text = textList[idx][offset_b, offset_e]
            labelEntryList.append((token_text, label, offset_b, offset_e))
        labelList.append(labelEntryList)

    # return format should be like [('text', 'class', [('token', 'label', 'offset_b', 'offset_e')]]
    result = zip(textList, classesTextList, labelList)
    print result

    return result