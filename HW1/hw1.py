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

        attributes = set(attributes[:-1])
        return attributes, attributeValues, label, datajson

    def predictLeafNode(self, data):
        p = 0
        n = 0
        for d in data:
            if d[self.label] == 'Yes':
                p += 1
            else:
                n += 1
        return 'Yes' if p >= n else 'No'

    def dataEntropy(self, currentData):
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

    def build(self):
        #Line 4:  Create a root node for the tree
        self.root = DecisionTreeNode(None, None, self.data)
        self.root = self.buildhelper(self.root, self.attributes)
        self.printTree(self.root)
        return

    def buildhelper(self, node, remainingAttributes):
        if(node.data is None):
            return None
        setEntropy = self.dataEntropy(node.data)

        if(len(remainingAttributes) == 0):
            node.chosenAttribute = self.label

            node.prediction = self.predictLeafNode(node.data)
            return node

        if setEntropy == 2:
            node.chosenAttribute = self.label
            node.prediction = 'Yes'
            return node
        if setEntropy == -2:
            node.chosenAttribute = self.label
            node.prediction = 'No'
            return node
        
        if setEntropy == 0 or len(node.data[0]) - 1 == 0:
            node.chosenAttribute = self.label

            node.prediction = self.predictLeafNode(node.data)
            return node

        bestAttribute, newRemainingAttributes = self.findHighestInformationGain(node, setEntropy, remainingAttributes)

        attributeSet = self.attributeValues[bestAttribute]
        node.chosenAttribute = bestAttribute
        for key in attributeSet:
            splitdata = []
            for line in node.data:
                if(line[bestAttribute] == key):
                    temp = copy.deepcopy(line)
                    del temp[bestAttribute]
                    splitdata.append(temp)
            if(len(splitdata) != 0):
                childNode = DecisionTreeNode(None, key, splitdata)
                newChildNode = self.buildhelper(childNode, newRemainingAttributes)
                if newChildNode is not None:
                    node.children.append(newChildNode)
                else:
                    node.prediction = self.predictLeafNode(node.data)
        return node

    def findHighestInformationGain(self, node, currentEntropy, remainingAttributes):
        maxGain = -1
        bestAttr =  None
        for currAttr in remainingAttributes:
                attrEntropy = self.findAttributeEntropy(node.data, currAttr)
                if currentEntropy - attrEntropy > maxGain:
                    maxGain = currentEntropy - attrEntropy
                    bestAttr = currAttr

        newRemainingAttributes = copy.deepcopy(remainingAttributes)
        newRemainingAttributes.remove(bestAttr)
        return bestAttr, newRemainingAttributes

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

    def printTree(self, node=None, indent=0):
        tabbing = '    ' 
        print(indent * tabbing, end = " ")
        if(node.prevAttributeValue):
            print(node.prevAttributeValue + ": ", end = " ")
        print(node.chosenAttribute)
        for child in node.children:
            self.printTree(child, indent + 1)
            if child.prediction:
                print((indent + 2) * tabbing, end = " ")
                print(child.prediction) 

    def predictQuery(self, query, currNode):
        if currNode.prediction:
            return "Based on the data " + str(query) + " you will " + ("not" if currNode.prediction == 'No' else '') + " enjoy"
        searchQuery = currNode.chosenAttribute
        searchQueryValue = query[searchQuery]
        nextNode = None
        for child in currNode.children:
            if child.prevAttributeValue == searchQueryValue:
                nextNode = child
                break
        if nextNode:
            return self.predictQuery(query, nextNode)
        else:
            temp = DecisionTreeNode(None, None, None)
            temp.prediction = self.predictLeafNode(currNode.data)
            return self.predictQuery(query, temp)



data = open("./dt_data.txt", 'r')
dt = DecisionTree(data)
predicitionQuery = {'Occupied': 'Moderate', 'Price': 'Cheap', 'Music': 'Loud', 'Location': ' City-Center', 'VIP': 'No', 'Favorite Beer': 'No'}
print(dt.predictQuery(predicitionQuery, dt.root))