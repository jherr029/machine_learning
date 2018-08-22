import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import pandas as pd
import collections
from math import sqrt
from sys import argv

arg = argv[1]
arg2 = argv[2]
overallCount = 0
rowMax = 69


# Converts to missing data to the average attribute
#   value of that particular class ( class 4 or class 2)
# Average value is then rounded to go with the rest of the data
#   since the are all integers
def convertQ(myData):
    print('\nconvert unknowns function')

    numOfAttr = len(myData)
    numOfQ = []
    colWithQ = []
    indicesWithQ = []

    for i in range(numOfAttr):
        temp = np.count_nonzero(myData[i] == '?')

        if temp > 0:
            colWithQ.append(i)
            numOfQ.append(temp)

    cnt = 0

    twoCnt = 0
    fourCnt = 0

    sumOfTwo = 0
    sumOfFour = 0

    for row, cur in zip(myData[5], range(len(myData[5]))):
        if row == '?':
            indicesWithQ.append(cur)
            cnt += 1
        else:
            if myData[9][cur] == '2':
                sumOfTwo += float(row)
                twoCnt += 1

            elif myData[9][cur] == '4':
                sumOfFour += float(row)
                fourCnt += 1

    avgTwo = round(sumOfTwo / twoCnt)
    avgFour = round(sumOfFour / fourCnt)

    for index in indicesWithQ:
        if myData[9][index] == '2':
            myData[5][index] = avgTwo

        if myData[9][index] == '4':
            myData[5][index] = avgFour

    return np.asfarray(myData, int)


#   Creates a panda data frame
def createPandasFrame(myData):
    print('\ncreate panda data frame function')

    # Creating a ordered dictionary
    # Use of label and its values
    # Needed for panda data frame
    od = collections.OrderedDict()
    od['Clump Thickness'] = myData[0]
    od['Uniformity of Cell Size'] = myData[1]
    od['Uniformity of Cell Shape'] = myData[2]
    od['Marginal Adhesion'] = myData[3]
    od['Single Epithelial Cell'] = myData[4]
    od['Bare Nuclei'] = myData[5]
    od['Bland Chromatin'] = myData[6]
    od['Normal Nucleoli'] = myData[7]
    od['Mitosis'] = myData[8]
    od['Class'] = myData[9]

    # Creating a panda data frame from the processed data matrix
    procData = pd.DataFrame(od)

    # Disacarding the last 9 rows
    procData = procData[:690]

    return procData


#   Splits the data from the data frame
#   10 splits each containing 69 data sets
def splitData(dataFrame):
    print('\nSplit data function')

    dfList = []
    #rowMax = 69

    #   Shuffles - removed at TA's request
    # TODO: take of comment
    # TODO: renable shuffling when not 2080
    if arg != '2080':
        dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

    # Makes the last 138 lines the test
    if arg == '2080':
        # TODO: originall 9
        for i in range(2):
            if i == 0:
                # print(dataFrame[-138:])
                dfList.append(dataFrame[-138:])
                dataFrame.drop(dataFrame.index[-138:], inplace=True)

            else:
                dfList.append(dataFrame[:])
                # print(dataFrame[:])
                dataFrame.drop(dataFrame.index[:], inplace=True)

        # print(len(dfList[0]))
        # print(len(dfList[8]))

    # Splits the data from the data frame 69 rows at a time
    else:
        for _i in range(10):
            dfList.append(dataFrame[:rowMax])
            dataFrame.drop(dataFrame.index[:rowMax], inplace=True)
            #dataFrame = dataFrame.reset_index(drop=True)

    if arg == 'split':
        print(dfList[0])

    return dfList


# Creates 10 folds to allow
# each part to be a test
def setupData(dataList):
    print('\nsetup Data function')

    if arg == '2080':
        rangeNum = 1

    elif arg2 == '1':
        rangeNum = 1

    else:
        rangeNum = 10


    # Creating a copy of the data
    tempList = dataList.copy()

    #   testList - A list of dataframes
    testList = []

    #   trainList - A list of of list of dataframes
    trainList = []

    #   Responsible for folding
    for cnt in range(rangeNum):
        # pops a section of the data to be the testSet
        testSet = tempList.pop(cnt)

        # the remaining parts are now the training set
        train = pd.concat(tempList)

        testList.append(testSet)
        trainList.append(train)

        #   Reset tempList
        #   copy is used in order to not actually
        #   delete the master list
        tempList = dataList.copy()

    if arg == 'setup':
        print('testList length:', len(testList))
        print('trainList length', len(trainList))
        print('\tinner trainList length', len(trainList[0]))

    # Get the labels from the trainList
    labelList = getLabels(trainList)

    # print(testList[1])

    # Get the labels from the testlist
    y_act = getLabels(testList)


    return (testList, trainList, labelList, y_act)

# Getting the labels from the parameter
def getLabels(x_train):
    print('\nget labels function')

    outList = []

    for outerList in x_train:
        outList.append(outerList.Class)


    if arg == 'label':
        #print('\tNum of outer list', numOfOuterList, end=' ')
        #print('Num of inner list', numOfInnerList)
        print()
        # print(x_train[8][8].Class)
        # print(outList[8][8])
    # print(x_train[0][0].Class)

    return outList


def checkTest(foldList, sortedList):
    # Folding list contains all the distances values
    # Folding list has the following structure
    # list[foldingNumber][trainingMatrix][testRow][distanceValue]

    # TODO: fix my notes
    # TODO: the labels

    for fold in sortedList:
        cnt = 0

        for trainM in fold:
            print(cnt)
            for testR in trainM:
                for i in range(10):
                    print(testR[i], end=' ')

                print()

            cnt += 1
            print('*'*60)
        print('!'*60)

    '''
    print(foldList[0][0][0])
    print()
    print(sortedList[0][0][0])
    '''


# Responsoble for getting the shortest distance aka the nearest
# neigbors
# Within this function all folds are taken care of
def knn_classifier_helper(x_test, x_train, y_train, k, p):
    print('\nknn classifier helper function')

    #distMatrix = np.zeros((69,9))
    foldList = []
    sortedFoldList = []

    # x_test - 10 -> 69
    # x_train - 10 -> 9 -> 69
    # y_train - 10 -> 9 -> 69

    # Iterates through each fold
    # One by one the folds are taken care of
    # From here shortest dostamce os calcualted and sorted
    for testListEle, trainListEle, labelListEle in zip(x_test, x_train, y_train):
        if arg == 'dist1':
            #print(j, ' --> ', end=' ')
            print()

        tempVec, sortedTempVec = iminkowski(
            testListEle, trainListEle, labelListEle, p)
        foldList.append(tempVec)
        sortedFoldList.append(sortedTempVec)

    if arg == 'check':
        checkTest(foldList, sortedFoldList)

    # Determiens the majority label woithin the shorteset distances
    # This is determined at all k values; 1  - 10
    y_pred = determineMajority(sortedFoldList)

    return y_pred
    # getNeighbors(foldList)

# Within this function minkowski distance is calculated
# It goes from the first row of the test set and calculate
# the distance of every row within the train set
# The same is repeated until each row of test set is taken care of
# This function is responsible for getting the arguments used to
# calculate the distance
def iminkowski(a, b, c, p):
    print('\niminkowski function')

    if arg == 'kow':
        print('\tA length', len(a))
        print('\tB length', len(b))
        for vc, i in zip(b, range(9)):
            print('\t', end=' ')
            for j in range(i):
                print(end=' ')
            print(len(vc))

    # TODO: need to iterate through the rows of a too??
    aLimit = len(a)          # 69 if 20/80 split then its 138
    bLimit = len(b)          # 9

    # TODO: delete
    #bInnerLimit = len(b[0])  # 69

    trainingDistanceList = []       # Outter most list - goes through the rows of A
    distancesList = []              # Middle list - size of 9 - # of matrix
    distancePointList = []          # Inner most list - goes the rows of B

    sortedDistances = []
    sortedTrainingDistances = []

    # So 69 - 69x1 Vectors for just A[0]
    # So overall 69 * 10 * 9 = ~450,000 Vectors ?!?!?

    # At each row of A get the distance of the first matrix of B

    outterCalls = 0
    middleCalls = 0
    numOfminCalls = 0


    # Here we work at the row level
    # TODO: change rnge to bLimit
    # TODO: swap the first two for loops
    # TODO: change the var of matrix to something better
    for aRow in range(aLimit):

        # TODO: change the range to aLimit
        for bRow in range(bLimit):

            # TODO: move this above
            aData = a.iloc[aRow]
            bData = b.iloc[bRow]
            cLabel = c.iloc[bRow]

            distValue = minkowski(aData, bData, p)

            # Tuples the distance value and the label it
            # is associated to
            dataTuple = (distValue, cLabel)

            distancePointList.append(dataTuple)

            numOfminCalls += 1
            #global overallCount

            #overallCount += 1

        temp = distancePointList
        # Sort the list
        temp = sorted(temp, key=lambda x: x[0])

        distancesList.append(distancePointList)
        sortedDistances.append(temp)


        distancePointList = []   # Empty out the list
        #trainingDistanceList.append(distancesList)
        #sortedTrainingDistances.append(sortedDistances)

        #distancesList = []    # Empty out the list
        #sortedDistances = []
        # print('*'*60)

    if arg == 'dist1':
        print('training distance list length:', len(trainingDistanceList))
        print('inside first index length:', len(trainingDistanceList[0]))
        print('inside inside first index length:',
              len(trainingDistanceList[0][0]))
        print('num of outter calls:', outterCalls)
        print('num of middle calls:', middleCalls)
        print('num of calls: ', numOfminCalls)
        print()


    return distancesList, sortedDistances


# Calculating the distances
def minkowski(X, Y, p):
    #print('\nminkowski function')

    xyList = []

    numOfAttr = len(X) - 1

    for xElement, yElement, _i in zip(X, Y, range(9)):
        xyElement = abs(xElement - yElement)
        xyElement = pow(xyElement, p)
        xyList.append(xyElement)

    xySum = sum(xyList)

    if p == 2:
        xySum = sqrt(xySum)

    if arg == 'dist2':
        print('length of X', len(X))
        print('length of Y', len(Y))
        print('Num of attr:', numOfAttr)
        print('xySum:', xySum)

    return xySum


# Determines the majority label from the list
# of knn neighbors from 1 - 10
# There is a special case for how
# Tied amount of labels are resolved
def determineMajority(foldList):
    print('\ndetermineMajority')

    outerList = []
    middleList = []
    knnList = []
    tempList = []

    cnt = 0
    tmpCnt = 0


    # Iterate through the fold list
    for fold in foldList:

        # Iterates through each row within the
        # specific fold
        # In this f
        for trainR in fold:

            classTwoCnt = 0
            classFourCnt = 0

            # Within this section it is determining
            # the majority label is determined at every
            # K level
            for i in range(10):
                temp = trainR[i][1]

                if temp == 2:
                    classTwoCnt += 1

                elif temp == 4:
                    classFourCnt += 1

                if classTwoCnt > classFourCnt:
                    classType = 2

                elif classTwoCnt < classFourCnt:
                    classType = 4

                else:
                    # If there is a tie, the majority is
                    # determined a different way
                    classType = tieBreaker(trainR, i + 1)

                tempList.append(classType)

                if arg == 'majority':
                    print(tempList, '-->', classType)

            #print()
            cnt += 1

            if arg == 'majority':
                print()

            knnList.append(tempList)
            tempList = []

        middleList.append(knnList)
        knnList = []

    if arg == 'majority':
        print(len(outerList))
        print(len(outerList[0]))
        print(len(outerList[0][0]))

    return middleList

# If there is a tie for majority label
# This part goes through the lsit
# and counts the distance for each label
# It returns whichoever of the two that has
# the shortest distance
# If there is a tie, just return 4
# ( its better to say they have cancer and
# be wrong about it)
def tieBreaker(data, max):
    #print('\ttie breaker function')

    distTwo = 0
    distFour = 0

    # break the tie
    for tple, _i in zip(data, range(max)):
        classType = tple[1]
        classDist = tple[0]

        if classType == 2:
            distTwo += classDist

        elif classType == 4:
            distFour += classDist

    if arg == 'tie':
        print('2:', distTwo, '4:', distFour)

    if distTwo > distFour:
        smallestClass = 4

    elif distTwo < distFour:
        smallestClass = 2

    else:
        # TODO: this needs to be resolved
        #print('TIE AGAIN')
        smallestClass = 4

    if arg == 'tie':
        print(smallestClass)

    return smallestClass

# debugging
def test(pred, act):
    print('\nTest')

    print(pred[0][0])

# This function is responsible for
# creating the confusion matrix
# in order to achieve this
# true positive, true negative
# false positive and false negative
# needs to be calculated
# within this function calcMetrics
# is called which calculates performance
def pos_neg(pred, act):
    print('\npos_neg')

    cnt = 0

    matrixInnerList = []
    matrixList = []
    knnList = []

    sizeOfKnn = len(pred[0][0])

    # Iterates through each fold
    for foldPred, foldAct in zip(pred, act):


        # Iterates through each knn
        for kNum in range(sizeOfKnn):

            tp = 0
            fp = 0
            tn = 0
            fn = 0

            #print(kNum)
            #print(foldAct)
            temp = np.zeros((2, 2))

            # Comparing the actual label to the predicted label
            # All comparisons are done at the k level
            for valPredList, valAct in zip(foldPred, foldAct):
                cnt += 1

                valPred = valPredList[kNum]

                #print(valAct, '-', valPred, end=' ')
                # True positive
                if 4 == valAct == valPred:
                    tp += 1
                    #print('tp', tp)

                # False negative
                elif 4 == valAct and valAct != valPred:
                    fn += 1
                    #print('fn', fn)

                # False positive
                elif 2 == valAct and valAct != valPred:
                    fp += 1
                    #print('fp', fp)

                # True negative
                elif 2 == valAct == valPred:
                    tn += 1
                    #print('tn', tn)

            temp[0][0] = tp
            temp[0][1] = fn
            temp[1][0] = fp
            temp[1][1] = tn

            # Calculating performances
            accuracy, sensitivity, specificity = calcMetrics(tp, fn, fp, tn)

            # Making a tuple of the confusion matrix and the
            # performances
            dataTuple = (temp, accuracy, sensitivity, specificity)

            #print(dataTuple[0])
            #print(dataTuple[1])
            #print(cnt)
            cnt = 0
            #print()

            knnList.append(dataTuple)

        matrixInnerList.append(knnList)
        knnList = []
                # print(matrixInnerList)


    #accList = calcAccuracy(matrixList)
    #accList = 0
    if arg == '2080':
        print('Confusion matrix')
        print(matrixInnerList[0][0][0])
        print('Accuracy')
        print(matrixInnerList[0][0][1])

    return matrixInnerList


# Calculates the performances
def calcMetrics(tp, fn, fp, tn):

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    #accuracy = float(('%.2f' % round(accuracy, 2)))
    accuracy = accuracy * 100
    accuracy = round(accuracy, 2)

    sensitivity = tp / (tp + fn)
    #sensitivity = float('%.2f' % round(sensitivity, 2))
    sensitivity = sensitivity * 100
    sensitivity = round(sensitivity, 2)

    specificity = tn / (tn + fp)
    #specificity = float('%.2f' % round(specificity, 2))
    specificity = specificity * 100
    specificity = round(specificity, 2)

    return accuracy, sensitivity, specificity


# This function calculates the statistics
# within all folds
# Mean and standard deviation
# This is done at each k level
# in all folds
# get a mean value and stdv at each fold
# for k
def calcStats(data):
    print('\nCalc stats')


    outList = []
    inList = []

    inMetricsList = []
    outMetricsList = []

    # Outer loop responsible for k
    for knn in range(10):
        #print('knn:', knn)

        accList = []
        senList = []
        speList = []

        # Goes through each fold
        for fold in range(len(data)):

            #print('  fold:', fold + 1)

            tempData = data[fold][knn]

            # Accuracy, Sensitivity, Specificity
            accList.append(tempData[1])
            senList.append(tempData[2])
            speList.append(tempData[3])

        #mean = sum(accList) / float(len(accList))
        accMean = round(st.mean(accList), 2)
        accStdv = round(st.stdev(accList), 2)

        senMean = round(st.mean(senList), 2)
        senStdv = round(st.stdev(senList), 2)

        speMean = round(st.mean(speList), 2)
        speStdv = round(st.stdev(speList), 2)

        accTuple = (accMean, accStdv)
        senTuple = (senMean, senStdv)
        speTuple = (speMean, speStdv)

        metricsTuple = (accList, senList, speList)

        statTuple = (accTuple, senTuple, speTuple)

        inList.append(statTuple)
        statTuple = []
        inMetricsList.append(metricsTuple)



        #print('   ', statTuple)
        #print('    ', metricsTuple)
        #outMetricsList.append(inMetricsList)

        #print('-'*35)

    #print(outList[0][0])
    return inList


# Creates the line graph for each performance
def createPlots(statList_1, statList_2):
    print('create plots')

    # Performance name
    metricType = ['Accuracy', 'Sensitivy', 'Specificity']

    # the x axix ticks
    xTicks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    #print(statList_1)
    p1List = []
    p2List = []

    p1Error = []
    p2Error = []

    # Outer loop to iterate through performance type
    for metricNum, metricName in zip(range(3), metricType):

        fileName =  metricName
        titleName = metricName

        # Get the mean and std at at k level at p1 ane p2
        # statList_1 is p 1 and the other is p 2
        for knn in range(10):
            metric_1 = statList_1[knn][metricNum][0]
            metric_2 = statList_2[knn][metricNum][0]

            error_1 = statList_1[knn][metricNum][1]
            error_2 = statList_2[knn][metricNum][1]

            p1List.append(metric_1)
            p2List.append(metric_2)

            p1Error.append(error_1)
            p2Error.append(error_2)


            #temp = temp.cumsum()

        # Creates a matrix in the required formate for pandas
        # to plot the the error bar
        err = getErrorBars(p1List, p2List, p1Error, p2Error)

        # Creating a data frame to be able to plot the data
        temp = pd.DataFrame(data=p1List, columns=['P - 1'])

        # Adding another column which contains the meanss of p2
        # at each k value at each performance
        temp['P - 2'] = p2List
        temp.index += 1
        #print(temp)
        #print(err)
        #print(np.shape(err))
        ax = temp.plot(title=titleName, xticks=xTicks, style='.-', yerr=err,
                        colormap='rainbow', grid=True, capsize=4)


        ax.set_xlabel('K-nn value')
        ax.set_ylabel(metricName + ' %')
        plt.savefig(fileName)
        plt.close()

        p1List = []
        p2List = []
        p1Error = []
        p2Error = []

# Uses standard deviation to get the error bars
# It is done in a specific formate that is required
# for pandas to olot the error bars
def getErrorBars(avg1, avg2, stdv1, stdv2):

    startList1 = []
    endList1 = []

    startList2 = []
    endList2 = []

    err = []

    for a1, s1 in zip(avg1, stdv1):
        start = round((s1), 2)
        end = round((s1), 2)

        #start = round((a1 - s1), 2)
        #end = round((a1 + s1), 2)

        startList1.append(start)
        endList1.append(end)

    err.append([startList1, endList1])

    for a2, s2 in zip(avg2, stdv2):
        start = round((s2), 2)
        end = round((s2), 2)

        #start = round((a2 - s2), 2)
        #end = round((a2 + s2), 2)

        startList2.append(start)
        endList2.append(end)

    err.append([startList2, endList2])

    return err



# Debugging
def printInfo(data):
    print('\nprint info')

    # TODO: delete the range
    for fold, i in zip(data, range(10)):
        print('Fold Number:', i + 1)

        for trainMatrix, j in zip(fold, range(9)):
            print(' Training Matrix:', j + 1)

            for knn, k in zip(trainMatrix, range(10)):
                print('  K Number', k + 1)

                #print(knn)
                print('  ', knn[0][0])
                print('  ', knn[0][1])
                print('    ', knn[1])
                print('    ', knn[2])
                print('    ', knn[3])
                print()

            print('-'*40)



def main(argv):
    print('main function')

    # Using unpack gets me the columns inteads of the line
    # Read from text file
    rawDataMatrix = np.loadtxt('cancer.data', unpack=True, dtype=str, delimiter=',',
                               usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    # Fills in the missing values
    procDataMatrix = convertQ(rawDataMatrix)

    # create a panda data frame
    pandaDataFrame = createPandasFrame(procDataMatrix)

    # Splits the data so that it will become foldable
    dataList = splitData(pandaDataFrame)

    # Creats the folds
    testList, trainList, labelList, y_act = setupData(dataList)

    # Gets the nearest neighbors
    if (arg != '2080'):
        y_pred_1 = knn_classifier_helper(testList, trainList, labelList, 1, 1)
        y_pred_2 = knn_classifier_helper(testList, trainList, labelList, 1, 2)

    else:
        y_pred_1 = knn_classifier_helper(testList, trainList, labelList, 1, 2)

    #test(y_pred, y_act)

    # Creates TP TN FP FN
    processedDataCube_1 = pos_neg(y_pred_1, y_act)

    if ( arg != '2080' ):
        processedDataCube_2 = pos_neg(y_pred_2, y_act)
        # calculates the mean and standard dev
        statList_1 = calcStats(processedDataCube_1)
        statList_2  = calcStats(processedDataCube_2)

        # Plots
        createPlots(statList_1, statList_2)



main(argv)
