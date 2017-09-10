# customized pipe data structure
#

class pipe:

    def __init__(self, text):
        if(text == None or (not text is str) or text == ''):
            raise ValueError("parameter text should be non-empty string, instead " + type(text))

        self.raw_text = text
        self.text_raw_unicode = unicode(text)
        self.entities = list()

    def getRowText(self):
        return self.raw_text

    def addEntity(self, token, entityType, offset_b, offset_a):
        my_entity = entity(token, entityType, offset_b, offset_a)
        self.entities.append(my_entity)

    def getAllEntities(self):
        return self.entities

    def setLabels(self, clazz, labels):
        if (clazz == None or (not clazz is str)):
            raise ValueError("parameter classes should be a list, instead " + type(clazz))
        if (labels == None or (not labels is [entity])):
            raise ValueError("parameter labels should be list<entity>, instead " + type(labels))
        self.clazz = clazz
        self.labels = labels

    def getClazz(self):
        return self.clazz

    def getLabels(self):
        return self.labels

    # TODO: return the orginal data. should be n*3 array. The 1st column is raw_text, the 2nd column is raw label with wd offsets
    def getDataToPredict(self):
        return ''

    # TODO: return converted data, for training. should be n*3 array. The 1st column is converted_text, the 2nd column classes, the 3rd column is type with wd offsets
    def getDataToTrain(self):
        # step 1: vectorize the list of words


        # step 2: for each word, asign label and entity


        # step 3: for entity, set the converted argute_entityType as the convertedText

        # step 4: if the scope of entity has label, set the first word convertedLabel

        # step 5: return result
        return ''

class entity:
    def __init__(self, token, entityType, label_wd_start, label_wd_end, label_char_start=None, label_char_end=None):
        self.token = token  # raw text
        self.entityType = entityType  # entityType
        self.et_char_start = label_char_start
        self.et_char_end = label_char_end
        self.et_wd_start = label_wd_start
        self.et_wd_end = label_wd_end
        self.source = None  # source of NER, not used yet

class response:
    def __init__(self, classes, entities):
        if(classes == None or (not classes is [resp_class])):
            raise ValueError("parameter classes should be list<resp_class>, instead " + type(classes))
        if (classes == None or (not entities is [resp_entity])):
            raise ValueError("parameter entities should be list<resp_entity>, instead " + type(entities))
        self.classes = classes
        self.entities = entities

class resp_class:
    def __init__(self, clazz, prob):
        if(clazz == None or (not clazz is str) or clazz == ''):
            raise ValueError("clazz should be non-empty string")
        if (prob == None or (not prob is float) or prob <= 0):
            raise ValueError("prob should be non-empty positive float")
        self.clazz = clazz
        self.prob = prob

class resp_entity:
    def __init__(self, token, type, prob):
        if(token == None or (not token is str) or token == ''):
            raise ValueError("token should be non-empty string")
        if(type == None or (not type is str) or type == ''):
            raise ValueError("type should be non-empty string")
        if(prob == None or (not prob is float) or prob <=0 ):
            raise ValueError("prob should be non-empty positive float")
        self.token = token
        self.type = type
        self.prob = prob