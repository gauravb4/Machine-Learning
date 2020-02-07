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
import copy
import queue


class DecisionTreeNode:
    
    def __init__(self, attribute, prev, data):
        self.chosenAttribute = attribute
        self.prevAttributeValue = prev
        self.data = data
        self.prediction = None
        self.children = []

class DecisionTree:

    def __init__(self, file):
        self.attributes, self.attributeValues, self.label, self.data = self.parseData(file)
        self.usedAttributes = set()
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

        attributeValues = {}
        for attr in attributes:
            tset = set()
            for line in datajson:
                if(line[attr] not in tset):
                    tset.add(line[attr])
            attributeValues[attr] = tset
                
        return attributes, attributeValues, label, datajson

    def build(self):
        #Line 4:  Create a root node for the tree
        self.root = DecisionTreeNode(None, None, self.data)
        self.root = self.buildhelper(self.root)
        self.printTree()
        return

    def predict(self, data):
        p = 0
        n = 0
        print(data)
        for d in data:
            if d[self.label] == 'Yes':
                p += 1
            else:
                n += 1
        return 'Yes' if p >= n else 'No'
    
    def printTree(self, node=None, indent=0):
        tabbing = '    ' 
        if not node:
            node = self.root
        print(indent*tabbing, end = " ")
        if(node.prevAttributeValue):
            print(node.prevAttributeValue + ": ", end = " ")
        print(node.chosenAttribute)
        for child in node.children:
            self.printTree(child, indent + 1)
            print((indent+1)*tabbing, end = " ")
            #if node.prediction:
            print(node.prediction)
            #else:
            #    print("Somethings missing")

    def buildhelper(self, node):
        setEntropy = self.remainingDataEntropy(node.data)
        #Line 5: If all examples are positive, Return the single-node tree Root, with label = +.
        #Line 6: If all examples are negative, Return the single-node tree Root, with label = -.
        if setEntropy == 2:
            node.chosenAttribute = self.label
            node.prediction = 'Yes'
            return node
        if setEntropy == -2:
            node.chosenAttribute = self.label
            node.prediction = 'No'
            return node
        if setEntropy == 0 or len(node.data) <= 1:
            if(node.prevAttributeValue == 'Mahane-Yehuda'):
                print("value should be yes here")
            node.chosenAttribute = self.label
            node.prediction = self.predict(node.data)
            return node
        bestAttribute = self.findHighestInformationGain(node.data, setEntropy)
        if bestAttribute == None:
            node.chosenAttribute = self.label
            node.prediction = self.predict(node.data)
            return node

        attributeSet = self.attributeValues[bestAttribute]
        node.chosenAttribute = bestAttribute
        self.usedAttributes.add(bestAttribute)

        for key in attributeSet:
            splitdata = []
            for line in node.data:
                if(line[bestAttribute] == key):
                    print("removing " + key + " from data set")
                    print(line)
                    temp = copy.deepcopy(line)
                    del temp[bestAttribute]
                    splitdata.append(temp)
            print("new data set length ", end = " ")
            print(len(splitdata))      
            childNode = DecisionTreeNode(None, key, splitdata)
            childNode = self.buildhelper(childNode)
            if childNode is not None:
                node.children.append(childNode)
            else:
                node.prediction = self.predict(node.data)
                
        
        return node


    def remainingDataEntropy(self, currentData):
        positiveLabel = 0
        negativeLabel = 0
        for line in currentData:
            if line[self.label] == 'Yes':
                positiveLabel += 1
            else:
                negativeLabel += 1
        
        if positiveLabel == 0:
            return -2
        elif negativeLabel == 0:
            return 2
            
        postiveEntropy = (((-positiveLabel) / (positiveLabel + negativeLabel)) * math.log2((positiveLabel) / (positiveLabel + negativeLabel)))
        negativeEntropy = (((negativeLabel) / (positiveLabel + negativeLabel)) * math.log2((negativeLabel) / (positiveLabel + negativeLabel)))
        return postiveEntropy - negativeEntropy 

    def findHighestInformationGain(self, currentData, currentEntropy):
        maxGain = -1
        bestAttr =  None
        for currAttr in self.attributes[:-1]:
            if currAttr not in self.usedAttributes:
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

        entropy = []
        posKeys = attributePositiveValueMap.keys()
        negKeys = attributeNegativeValueMap.keys()
        keySet = set()
        for item in posKeys:
            keySet.add(item)
        for item in negKeys:
            keySet.add(item)
        for key in keySet:
            p = attributePositiveValueMap[key] if key in posKeys else 0
            n = attributeNegativeValueMap[key] if key in negKeys else 0
            if(p != 0 and n != 0):
                postiveEntropy = (((-p) / (p + n)) * math.log2((p) / (p + n)))
                negativeEntropy = (((n) / (p + n)) * math.log2((n) / (p + n)))
                entropy.append(((p + n) /  len(currentData)) * (postiveEntropy - negativeEntropy))
            else:
                entropy.append(0)

        return sum(entropy)
        

data = open("./dt_data.txt", 'r')
dt = DecisionTree(data)
