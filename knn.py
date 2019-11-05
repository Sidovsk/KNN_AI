import csv
import math
import numpy

def cosineSim(sample, data):
    cosines = []
    for row in data:
        numeratorSum = 0
        sampleNorm = 0
        rowNorm = 0
        for i in range(0, len(sample)):
            numeratorSum += sample[i]*row[i]
            sampleNorm += sample[i]**2
            rowNorm += row[i]**2
        sampleNorm = math.sqrt(sampleNorm)
        rowNorm = math.sqrt(rowNorm)
        cosine = numeratorSum/(sampleNorm*rowNorm)
        cosines.append(cosine)
    labeledCosines = []
    for i in range(len(cosines)):
        labeledCosines.append((cosines[i], data[i][-1]))
    labeledCosines.sort(key=lambda tup: tup[0], reverse= True)
    return labeledCosines

def euclideanSim(sample, data):
    distances = []
    for row in data:
        sum = 0
        for i in range(0, len(sample)):
           sum += (sample[i] - row[i])**2
        sqr = math.sqrt(sum)
        distances.append(sqr)
    labeledDistances = []
    for i in range(len(distances)):
        labeledDistances.append((distances[i], data[i][-1]))
    labeledDistances.sort(key=lambda tup: tup[0])
    return labeledDistances

def knn(k, data, sample, simMeasure):
    #calculating similarity
    simArray = simMeasure(sample, data)
    #manage classification
    classesArray = []
    for i in range(k):
        newClass = True
        for j in range(len(classesArray)):
            if(simArray[i][1] == classesArray[j][0]):
                newClass = False
                classesArray[j] = (classesArray[j][0], classesArray[j][1]+1) #increment quantity
        if(newClass):
            classesArray.append((simArray[i][1], 1)) #making a array of tuples with the pair (class, quantityOfNeighbors)
    #get class with more neighbors
    classesArray.sort(key=lambda tup: tup[1], reverse=True)
    print(classesArray)
    return classesArray[0][0] #return the k nearest neighbor
    
with open('./spambase.data') as f:
    csvData = csv.reader(f)
    next(csvData)
    data = []
    for row in csvData:
        row = row[1:]
        for i in range(len(row)-1):
            row[i] = float(row[i])
        data.append(row)
    numpy.random.shuffle(data)
    #separating 75% to train and 25% to test
    trainPercent = int(0.75*len(data))
    train, tests = data[:trainPercent], data[trainPercent:]
    #Measuring accuracies
    hits = 0
    for test in tests:
        if(knn(3, data, test[:-1], cosineSim) == test[-1]):
            hits += 1
    print(hits/(len(tests)))