'''
Following ID3 Algorithm on Wikipedia
ID3 (Examples, Target_Attribute, Attributes)
    Create a root node for the tree
    If all examples are positive, Return the single-node tree Root, with label = +.
    If all examples are negative, Return the single-node tree Root, with label = -.
    If number of predicting attributes is empty, then Return the single node tree Root,
    with label = most common value of the target attribute in the examples.
    Otherwise Begin
        A ← The Attribute that best classifies examples.
        Decision Tree attribute for Root = A.
        For each possible value, vi, of A,
            Add a new tree branch below Root, corresponding to the test A = vi.
            Let Examples(vi) be the subset of examples that have the value vi for A
            If Examples(vi) is empty
                Then below this new branch add a leaf node with label = most common target value in the examples
            Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
    End
    Return Root
'''
import math


class DecisionTreeNode:
    
    def __init__(self, attribute, prev, data):
        self.attribute = attribute
        self.prev = prev
        self.data = data
        self.answer = None
        self.children = []
        self.choices = {}

class DecisionTree:

    def __init__(self, file):
        self.attributes, self.label, self.data = self.parseData(file)
        #print(self.data)
        self.build()
        return

    def parseData(self, file):
        f = file.readlines()
        attributes = f[0][1:-2].strip().split(", ")
        label = attributes[-1]
        attribute_data = f[2:]
        attribute_values = [line[4:line.index(';')] for line in attribute_data]
        attribute_values = [line.split(", ")for line in attribute_values]
        datajson = []
        
        data = {}
        for line in attribute_values:
            data = {}
            for i, attribute in enumerate(attributes): 
                data[attribute] = line[i]

            datajson.append(data)

        return attributes, label, datajson

    def build(self):
        #Line 4:  Create a root node for the tree
        self.root = DecisionTreeNode(None, None, self.data)
        self.buildhelper(self.root)
        return

    def buildhelper(self, node):
        if node.data is None:
            return None
        setEntropy = self.remainingDataEntropy(node.data)
        #Line 5: If all examples are positive, Return the single-node tree Root, with label = +.
        #Line 6: If all examples are negative, Return the single-node tree Root, with label = -.
        if setEntropy == 0:
            return None
        
        print(setEntropy)
        bestAttribute = self.findHighestInformationGain(node.data, setEntropy)
        print(bestAttribute)

        attributeIndex = self.attributes.index(bestAttribute)
        

        return None


    def remainingDataEntropy(self, currentData):
        positiveLabel = 0
        negativeLabel = 0
        for line in currentData:
            if line[self.label] == 'Yes':
                positiveLabel += 1
            else:
                negativeLabel += 1
        
        if positiveLabel == 0:
            return 0
        elif negativeLabel == 0:
            return 0

        postiveEntropy = (((-positiveLabel) / (positiveLabel + negativeLabel)) * math.log2((positiveLabel) / (positiveLabel + negativeLabel)))
        negativeEntropy = (((negativeLabel) / (positiveLabel + negativeLabel)) * math.log2((negativeLabel) / (positiveLabel + negativeLabel)))
        return postiveEntropy - negativeEntropy 


    def findHighestInformationGain(self, currentData, currentEntropy):
        maxGain = -1
        bestAttr =  None
        for currAttr in self.attributes[:-1]:
            attrEntropy = self.findAttributeEntropy(currentData, currAttr)
            if currentEntropy - attrEntropy > maxGain:
                maxGain = currentEntropy - attrEntropy
                bestAttr = currAttr

        return bestAttr

    def findAttributeEntropy(self, currentData, currAttr):
        attributePositiveValueMap = {}
        attributeNegativeValueMap = {}
        for line in currentData:
            if (line[self.label] == 'Yes'):
                if line[currAttr] in attributePositiveValueMap:
                    attributePositiveValueMap[line[currAttr]] = attributePositiveValueMap[line[currAttr]] + 1
                else:
                    attributePositiveValueMap[line[currAttr]] = 1
            else:
                if line[currAttr] in attributeNegativeValueMap:
                    attributeNegativeValueMap[line[currAttr]] = attributeNegativeValueMap[line[currAttr]] + 1
                else:
                    attributeNegativeValueMap[line[currAttr]] = 1
        #print(attributeNegativeValueMap)
        #print(attributePositiveValueMap)

        entropy = []
        for key in attributePositiveValueMap:
            p = attributePositiveValueMap[key]
            n = attributeNegativeValueMap[key]
            postiveEntropy = (((-p) / (p + n)) * math.log2((p) / (p + n)))
            negativeEntropy = (((n) / (p + n)) * math.log2((n) / (p + n)))
            entropy.append(((p + n) /  len(currentData)) * (postiveEntropy - negativeEntropy))

        return sum(entropy)
        


    

data = open("./dt_data.txt", 'r')
dt = DecisionTree(data)
