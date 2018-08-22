import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import argv
import sys
import math
import statistics as st


# ignore this
#   Arguements
# printCentroids - prints centroid values
# clusterChanges - prints the centroid chang
# work - prints the clusters points
# distances - prints the distance values
# meanChanges - prints the changes 
# checkMatch - check if distances match to previous



# Read from the file
def readFile():
    print('\ncreate data frame')
    dataFrame = pd.read_csv('iris.data', header=None, sep=',')
    dataFrame.columns = ['sepal length', 'sepal width', 'petal length', 
                            'petal width', 'class']

    dataFrame.index += 1

    return dataFrame


# initailizing centroids 
# gets random data point to be
# a centroid
# If two data points are similar
# recursively calls itself to get new
# data points
def initCentroids(df, k):
    print('\ninit Centroids')

    # get k data points
    init = df.sample(n=k)

    # boolean to declare that all
    # data points in init are unique
    uniqueFlag = True

    # converts the data frame to a matrix
    matrix = init.as_matrix()
    print(matrix)

    # get the length of the matrix
    matLen = len(matrix)

    cnt = 0
    rowA = 0
    rowB = 0
    
    # If k is 1 then the sample is unique
    # else check that the sample contains
    # unique data points
    if k != 1:
        for cur in range(matLen):
            for comp in range(cur, matLen):
                if cur != comp:
                    if np.all(matrix[cur] == matrix[comp]):
                        uniqueFlag = False
                        rowA = cur + 1
                        rowB = comp + 1
                        cnt += 1

    if uniqueFlag == False:
        print('Theres a matching pair')
        print(cnt)
        print(rowA, rowB)
        matrix = initCentroids(df, k)
    
    
    # Returns the randomly selected centroids
    return matrix
    

# Performs k means
# df - the dataFrame from the iris.data file (data points)
# k - the k value
# centroids - the centroid
# TODO: maybe do this recursively?
def k_means_cs171(df, k, centroids):
    print('\nk_means_cs171')

    # converts data frame to a matrix
    matrix = df.as_matrix()

    i = 0               # i is used as a iterator
    cont = True         # bool value used in while loop
    boolVal = False     # bool value to indicate if there has been
                        # any changes from the new and old cluster

    oldDistResult = []  # holds the old results (the old clusters)

    # Loop until there has been no new changes to the clustes
    while cont == True:
        # Not really needed since the while loop converse this
        #if i < 10000:

        #print('Loop:', i + 1)

        # debugging, prints centroids
        if 'printCentroids' in argv:
            print('centroids:\n', centroids)
        
        # getDistannce
        # caclualtes the distances between data points and centroids
        # returns two list
        # resultList - the data points and the cluster it belongs to
        # distResult - the distances of the data points to the centroids
        resultList, distResult = getDistances(matrix, k, centroids)

        # After the first loop go through this branch
        if i > 0:
            #print()
            #print(resultList)
            #print(oldResultList)

            # checks to see if the old results and the new results are similar
            # if they are similar then no new changes will occur
            # therefore it returns true
            # else false
            boolVal = checkIfMatch(oldDistResult, distResult)

        # if the result is unique
        if boolVal == False:

            # spliting the result list
            # formating my results (not essential part of k-mean algorithm)
            # sorts each cluster to be in its own index in the list
            nDimList = splitList(resultList, k)

            if 'clusterChanges' in argv:
                for cnt in range(k):
                    print(' ', len(nDimList[cnt]), 'in Cluster #', cnt + 1)


            # create new centroids based on the mean of all the data points
            # within the cluster
            centroids = createNewCentroids(nDimList, centroids)

            # place the result into old variable
            oldDistResult = distResult
        
        # if old result == new result
        # end the while loop
        elif boolVal == True:
            cont = False
        
        i += 1

        print('-'*45)
        
    # debugging statement
    if 'work' in argv:
        showClusters(nDimList, centroids)
    
    # returns the cluster and centroids
    return nDimList, centroids
    

# Calculates the distance
def getDistances(matrix, k, centroids):
    print('\nget distances')

    # TODO: delete distanceList
    kLevel = []
    resultList = []
    distanceList = []

    # gets length of the matrix
    matrixLength = len(matrix)

    # goes through every data point
    for row in range(matrixLength):

        # data point
        tempRow = matrix[row]
        #print(cnt, tempRow, end=' -> ')
        
        # iterates through the centroids
        for kNum in range(k):
            # euclidean distance
            distanceTemp = np.linalg.norm(tempRow-centroids[kNum])
            distanceTemp = round(distanceTemp, 4)

            #print(distanceTemp)
            #places it into a list
            kLevel.append(distanceTemp)


        # finds which centroid the data point is close to
        # and returns the index (cluster) that data point 
        # is close to
        index = pickCluster(kLevel, centroids)

        if 'distances' in argv:
            print(row + 1, kLevel, ' ', index + 1)

        # tuples
        tempTuple = (tempRow, index) # data point and cluster it belongs to
        disTuple = (kLevel, index)   # data point distance to centroid and centroid
                                     # it belongs to
        
        # placing each truple to its respective list
        resultList.append(tempTuple)
        distanceList.append(disTuple)

        # reseting the list
        kLevel = []

    return resultList, distanceList

# gets the closest cluster of the data point
def pickCluster(distance, centroids):
    #print('pick cluster')

    # TODO: verifiy this
    # Picks the value that is the minimum
    smallest = min(distance)
    # Gets the index corresponding to the 
    # minimum value
    index = distance.index(min(distance))

    # that index value is the cluster
    return index


# This function is needed to separate the data point
# and the cluster it belongs to from each other
# the reason for this is because it is a tuple
# tuple - > (data point, cluster)
def splitList(distances, k):
    print('\nSplit List')

    # makes k list within a list
    tempList = [[] for NULL in range(k)]

    # a new list containting the clusters
    newList = []

    # Sort tempList with the clusters
    for value in distances:
        tempList[value[1]].append(value[0])

    # puts everything in a simplified list
    for i in range(len(tempList)):
        #print(i + 1)
        #print(tempList[i])

        # vstack changes the list and puts it in
        # array format vertically
        if len(tempList[i]) != 0:
            newList.append(np.vstack(tempList[i]))
        else:
            #print('YES')
            # this occurs when no point belongs to the cluster
            # appends an empty list
            newList.append([])
            #globalVar = True
        #print()
    
    # returns the cluster
    return newList


# creates new centroids based on the mean of all the
# data points within it
def createNewCentroids(nDimList, centroids):
    print('\nCreate new centroids')

    # the list that will contian the updated centroid values
    centroidList = []

    # goes through every k centroid
    for i in range(len(nDimList)):

        # if the centroid has no data points within it
        # use the same centroid value again
        if len(nDimList[i]) == 0:
            centroidList.append(np.asarray(centroids[i]))
        
        else:
            # get the mean value of the data point     
            meanResult = getMeans(nDimList[i], i)
            # append it to the list
            centroidList.append(np.asarray(meanResult))

    #print(centroidList)
    # stack every individual array to make it
    # one 2d array (vertically)
    centroidList = np.vstack(centroidList) 

    return centroidList

# gets the means of the data points
def getMeans(nCluster, i):

    temp = np.mean(nCluster, axis=0)
    temp = np.round(temp, 3)
    if 'meanChanges' in argv:
        print(' new centroid -', i + 1, temp)

    return temp


# check results of the previous run and the new run
# the new cluster changes and the old cluster changes
# if they are the same return false else return true
def checkIfMatch(old, new):
    print('Check if match')

    for i in range(len(old)):
        rslt = old[i][1] == new[i][1]
        if 'checkMatch' in argv:
            print(old[i][1], '-', new[i][1], rslt)
            print()

        if rslt == False:
            print(' False')
            return False
    
    print(' Complete!')
    return True


# a print function for debugging
def showClusters(clusterList, centroids):
    print('\nShow clusters')
    clusterNum = 0

    for clusterNum in range(len(clusterList)):
        listElement = clusterList[clusterNum]
        if len(listElement) != 0:

            print(clusterNum + 1, '|', centroids[clusterNum])

            for ele in listElement:
                print(' ', ele)
        
            print()
    
    for clusterNum in range(len(clusterList)):
        listElement = clusterList[clusterNum]
        if len(listElement) != 0:
            print('Cluster #', clusterNum + 1,  '---', len(listElement), 'points')
    
    print()


# sum of squared error calculation
# returns the sum of squared error in a list
# codeCheck by default will be false if the argument is
# not provided. That argument is for the grader 
# to check the sse values at every centroid
def sse(clusterData, centroids, codeCheck=False):
    print('sum of square errors calc\n')

    sumList = []
    sumError = 0

    # anything involving codeCheck is
    # for checking if the code works
    # for the graders
    if codeCheck == True:
        print('*'*60)

    # Goes through each cluster
    for clusterNum in range(len(clusterData)):

        cluster = clusterData[clusterNum]
        clusterCentroid = centroids[clusterNum]


        # TODO: Look into this
        if len(cluster) != 0:

            # goes through each data point in the cluster
            for dataIndex in range(len(cluster)):
                dataPoint = cluster[dataIndex]
                
                # data point minus its respective centroid
                diff = dataPoint - clusterCentroid
                diff = diff ** 2    # squares the result

                # getting the overall sum
                sumError += diff
            
        # rounds the result four decimal places
        sumError = np.round(sumError, 4)
        #sumList.append(sumError)

        if codeCheck == True:
            print('Code check printout')
            print('centroid:', clusterCentroid)
            print('sse:', sumError)
            print()


    # sums all the results together to get one indivual number
    resultError = np.sum(sumError)
    #print(resultError)

    if codeCheck == True:
        print('*'*60)

    # rounds the result to 2 decimal places
    return round(resultError, 2)


# gets a line graph which will look like a knee
# uses the sum of squared errors to fill in the graph
# this is for the normal (non sensitivity)
def createKnee1(sse):
    print('create knee plot')

    # the x axis that represents the k value
    xArr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # puts the sse list into a data frame to graph
    df = pd.DataFrame(sse)

    # increases the index of the data frame by one
    # not really needed
    df.index += 1

    # plots the graph
    ax = df.plot(title='Knee Plot', xticks=xArr, grid=True, color='green')

    # setting the axis labels
    ax.set_xlabel('K Cluster(s)')
    ax.set_ylabel('Sum Squared Error')

    plt.savefig('reg_knee')
    plt.close()

    #print(df)


# same thing as the above function put this is for the
# sensitivity result
# only difference is that this has 3 additional
# parameters
# stdev - standard deviation
# run - the max_itr number
# typeK - whether normal k means was used
#         or kmeans++
def createKnee2(sse, stdev, run, typeK):
    print('create knee plot')

    xArr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    df = pd.DataFrame(sse)
    df.index += 1

    # if normal set the title name
    # and the save file name
    if typeK == 'normal':
        titleName = ''
        saveName = ''
    
    # same as above
    elif typeK == 'plus':
        titleName = ' ++'
        saveName = '_plus'

    # plots the graph
    ax = df.plot(title=str(run) + ' Knee Plot' + titleName, xticks=xArr, grid=True, 
                    color='green', yerr=stdev, ecolor='red', capsize=8)
    
    # setting the axis labels
    ax.set_xlabel('K Cluster(s)')
    ax.set_ylabel('Sum Squared Error')

    plt.savefig(str(run) + '_knee' + saveName)
    plt.close()

    #print(df)


# performs k means but without sensitivity checking
def normal(x_input, codeCheck):
    sseList = []


    if codeCheck == True:
        kVal = 3
    
    else:
        kVal = 10

    # loop 10 times
    # Pretty much k 1 ... 10
    for k in range(kVal):
        k = k + 1

        # gets the initial centroids random
        init_centroids = initCentroids(x_input, k)

        # performs k-means
        # dataPoints - the data points sorted by cluster it belongs to
        # centroids - the final centroids for each cluster
        dataPoints, centroids = k_means_cs171(x_input, k, init_centroids)

        # gets the sum squared error

        sseList.append(sse(dataPoints, centroids, codeCheck))
        


    if codeCheck == False:
        # creates the line plot using the sum squared error
        createKnee1(sseList)    

    print('sse result')
    print(sseList)
    #print()
    

# the following follows the exact same logic as the above function
# the only difference is that this focuses on sensitivity
# it will loop 2, 10, and 100 times to check the sensitivity
def sens(x_input):

    # a list that contains the overall
    # sse values from 2, 10, and 100 runs
    masterSSElist = []

    #the max_itr sensitivity runs
    maxItrList = [2, 10, 100]

    # Access the maxItrList
    for maxItr in maxItrList:

        # loops 2, 10, 100 times
        for _i in range(maxItr):

            # a list to contain sseValues
            sseList = []

            # k iterations
            for k in range(10):
                k = k + 1

                # get k random centroids
                init_centroids = initCentroids(x_input, k)

                # k mean calculations
                dataPoints, centroids = k_means_cs171(x_input, k, init_centroids)

                # sse list
                sseList.append(sse(dataPoints, centroids))

            # master list for sse values
            masterSSElist.append(sseList)
            
            # get the means and standard deviations fron the sse values
            meanResult, stDevResult = getStats(masterSSElist)

        # plot the line graph
        createKnee2(meanResult, stDevResult, maxItr, 'normal')    
    
    return meanResult


# get the statistics of the sse values
# the mean and standard deviation
def getStats(sse):
    print('get stats')

    # format sse list into an array
    # in order to use numpy's mean and 
    # standard deviation
    sse = np.vstack(sse)

    # gets the mean
    meanResult = np.mean(sse, axis=0)
    # gets the standard devation
    stDevResult = np.std(sse, axis=0)

    if 'stats' in argv:
        print(sse)
        print(meanResult)
        print(stDevResult)

    # return mean and standard deviation
    return meanResult, stDevResult


# this is the k-means++ verstion
# the logic and layout is similar to the 2 above functions
# the only difference is that the initial centroids are not
# are computed differently
def kPlus(x_input):
    masterSSElist = []
    maxItrList = [2, 10, 100]

    matrix = x_input.as_matrix()

    for maxItr in maxItrList:

        for _i in range(maxItr):

            sseList = []

            for k in range(10):
                k = k + 1
                print('\n\t\tk value:', k)

                # gets the initial centroids differently
                init_centroids = initCentroidsPlus(x_input, k)

                print(init_centroids)

                dataPoints, centroids = k_means_cs171(x_input, k, init_centroids)

                sseList.append(sse(dataPoints, centroids))


            masterSSElist.append(sseList)

            meanResult, stDevResult = getStats(masterSSElist)
                
        createKnee2(meanResult, stDevResult, maxItr, 'plus')    
    
    return meanResult


# function responsible for getting the ++ centroids
def initCentroidsPlus(df, k):
    print('init centroids plus')

    centroids = initCentroids(df, 1)
    matrix = df.as_matrix()
    distList = []
    print('---'*6)

    # when k is one just get a random centroid
    if k == 1:
        centroids2 = centroids
        return centroids
    
    else:

        # if k is not 1, the centroids are obtained differently
        for k in range(2, k + 1):

            #print('????')
            # a list that will contain centroids
            # the list will get cleared at every loop iteration
            # the reason for using a list is because the getDistances
            # function works with a list
            centroidList = []

            # centroids[-1] is only accessing the last element in the list
            # the reason why were are putting only the last element in the list
            # is to prevent putting the 
            centroidList.append(centroids[-1])

            # get the distances from the data points to the centroid
            resultList, distResult = getDistances(matrix, 1, centroidList)

            # TODO: probably delete, it is being reused
            centroidList = []

            # squares the result of the distances
            # also gets the summations
            d2Tuple = squares(distResult)
            #TODO: replace that d2Sum line
            distList.append([i[0] for i in d2Tuple])

            #print(centroids[-1])

            # when working at k 3 and beyond a new step has to be added
            if k > 2:
                print('k is bigger than 2')

                # gets the minimum from either cluster at each data point
                # that weight is then used to get D2 (weight aka distances)
                # within this function the next centroid index is also determined
                centroidIndex = getMins(distList)

            else:

                # gets the centroid index 
                # usees the probability to get the new centroid
                centroidIndex = getCentroid(d2Tuple)

            # uses the centroidIndex to get the new centroid from the matrix
            # matrix contains all data points
            newCentroid = matrix[centroidIndex]

            # adds new centroids to the previous centroids
            centroids = np.vstack([centroids, newCentroid])
            #print(centroids)

    return centroids


# Squares the distance and get the sums
def squares(distList):

    sumResult = 0

    # list that contains a tuple that has distance^2 values
    # and the current summation
    d2List = []

    # Not really used
    fracList = []

    cnt = 0
    for dist in distList:

        # Square the distances
        # the reason for the accessing looking strange
        # is due to the value being inside a tuple
        temp = dist[0][0] ** 2

        # get the sum
        sumResult += temp

        # tuple the d^2 and the summation
        tempTuple = (temp, sumResult)
        #print(cnt, tempTuple)

        # put in a list
        d2List.append(tempTuple)
        cnt += 1

    return d2List


# get the new centroid (part of kmeans++)
def getCentroid(dataTuple):
    print('get centroid')

    # gets the distance value from the tupleb
    # and puts it into a list which gets converted
    # to a array the same occurs in the else if

    # the if is to check what data structure
    # is dataTuple in order to access the
    # contents correctly

    # checks if tuple
    if isinstance(dataTuple[0], tuple):
        d2Sum = [i[1] for i in dataTuple]
        d2Sum = np.array(d2Sum)
    
    # else if the data being accessed is a float64
    elif isinstance(dataTuple[0], np.float64):
        d2Sum = np.array(dataTuple)
        #print(d2Sum)

    # get the last number in the list of sums
    endNum = d2Sum[-1]
    #print('end num', endNum)

    # get a random number from 0 to endNum
    randNum = np.random.uniform(0, endNum)

    #print(randNum)

    # a helper function to reduce number of iterations 
    # in getting whatever summation value is close to the 
    # random number and return the index
    centroidIndex = getCentroidHelper(d2Sum, randNum)

    return centroidIndex


# I could have used a function to get whatever value
# is close to the random number but this should be a faster
# implementation
# since d2Sum is sorted I am able to work at different folds of
# of the d2Sum
# [0....74....149]
# if the random number is smaller than element 74 then work in left fold
# else work on the right fold
# Then [0....36....74]
# [75....111....149]
# then the same logic followed within the last folds
def getCentroidHelper(d2Sum, randNum):

    # [0....74]
    if randNum < d2Sum[74]:

        # [0....36]
        if randNum < d2Sum[36]:

            startIn = 0
            endIn = 36

        # [37....74]
        else:
            startIn = 37
            endIn = 74

    # [74....149]
    else:   

        # [75....111]
        if randNum < d2Sum[111]:

            startIn = 75
            endIn = 111
        
        # [112....149]
        else:

            startIn = 112
            endIn = 149
        
    # once the fold is located loop through the fold to get the cloeset
    # summation to that random number
    diffIndex = loopForMin(d2Sum, startIn, endIn, randNum)

    print('index value:', diffIndex)
    #print('diff', diff, 'index', diffIndex)

    return diffIndex



# a loop that iterates through the folds to get the
# sum value to the random number and returns the sum
def loopForMin(d2Sum, start, end, randNum):

    #diff = abs(d2Sum[start] - randNum)
    #diffIndex = start

    for i in range(start, end + 1):

        print(d2Sum[i] , i, '<', randNum, '<', d2Sum[i+1], i + 1)

        if 0 <= randNum and d2Sum[start] >= randNum:
            print('0 <', randNum, '<', d2Sum[start], start)
            print('STAAAART')
            return start

        if d2Sum[i] < randNum and d2Sum[i + 1] > randNum:
            print('TRUEEEE')
            return i + 1


# checks each centroid distance to the data points
# and gets the one with the minimum distance value
# once that is obtained get the summation at every
# iteration of the run
def getMins(distList):

    # need to see if this really works
    '''
    for i in range(150):
        for cluster in range(len(distList)):
            print(distList[cluster][i], end=' - ')

        print()
    '''
    
    #minVal = min(distList, key=lambda x: x[0])
    # gets the mininum distance value like mentioned
    # in the function description
    minVal = getMinHelper(distList)
    #print()
    #print(minVal)

    # gets the summation at every level and puts it in the list
    # the summation is from every value in minVal
    totalSum = 0
    sumList = []
    for val in minVal:
        totalSum += val
        sumList.append(totalSum)


    # gets the centroid based on a random number
    # and whatever data point is close to that
    # random number
    centroidIndex = getCentroid(sumList)

    #tempTuple = (distList[-1], sumList)
    #print([i[1] for i in tempTuple])

    return centroidIndex


# iterates through each cluster at a row level
# and places smallest distance in a list
# the list with the smallest distance is returned
def getMinHelper(distList):

    clustNum = len(distList)
    print('clustNum', clustNum)
    #print(distList[0])
    numRow = 150
    #minNum = 99999
    index = 999

    minList = []

    # iterates through the data points distance
    for row in range(numRow):

        minNum = 99999

        # iterates through each cluster distance
        # to the data point
        for clust in range(clustNum):
            temp = distList[clust][row]
            #print(temp)

            # gets the smallest distance
            if temp < minNum:
                minNum = temp
                index = clust
        
        # puts smallest distance in a list
        minList.append(minNum)
    
    minArr = np.asarray(minList)

    return minArr


def topThree(x_input, df):

    # gets three random data points to be centroids
    init_centroids = initCentroids(x_input, 3)

    # performs k means with k of three
    dataPoints, centroids = k_means_cs171(x_input, 3, init_centroids)

    # gets the top 3 data points within each cluster relative to its
    # centroid
    closestPoints = getClosests(dataPoints, centroids)

    # gets the label of the top 3 data points of each cluster
    getLabels(closestPoints, centroids, x_input, df)


# function that gets the closests three data points
# to its respective centroid
def getClosests(clusterField, centroids):
    print('get closests')
    
    clusterMinList = []

    # iterates through each cluster and its data point set
    for centroid, cluster in zip(centroids, clusterField):

        distList = []
        minList = []

        #rint('centroid', centroid)

        # get the distances of the data points to its cluster
        for point in cluster:
            distanceTemp = np.linalg.norm(point-centroid)
            distList.append(distanceTemp)
            #print(point)
        #print('---'*15)

        # iteratees 3 times to get the 3 closest points
        for i in range(3):

            cnt = 0
            minVal = 9999
            minValIndex = 0

            # goes through all the points to find 
            for dist in distList:
                #remove from list del distList[]
                #print(dist)
                if dist < minVal:
                    minVal = dist
                    minValIndex = cnt
                
                cnt += 1
            
            # to avoid using the same value overwrite its content
            # with 9999
            distList[minValIndex] = 9999
            tempTuple = (cluster[minValIndex], minVal)
            minList.append(tempTuple)
        
        clusterMinList.append(minList)
    
    #print('look')

    resultList = []

    # detuples the data
    for top3 in clusterMinList:
        #print(top3)

        temp = [i[0] for i in top3]
        temp = np.asarray(temp)
        resultList.append(temp)
        #print(temp)

        #print('*'*30)
    
    # returns the top 3 data points within each cluster
    return resultList


# gets the labels of each top point in its cluster
def getLabels(cluster, centroids, matrix, df):
    # gets only the class column of the data frame
    df = df['class']

    matrix = matrix.as_matrix()

    # iterates through the cluster
    for pointSet, centroid in zip(cluster, centroids):
        print('centroid:', centroid)

        # goes through the top 3 points
        for point in pointSet:

            # gow through the all the data points 
            # to able to get the label
            for index, row in enumerate(matrix):

                # then this is the matching data point
                # even though there are data points that have
                # the same values they are of the same class type
                # so I ignored getting the actual correct data point
                if np.all(point == row):
                    print(point, ' - Row ', index + 1, ' - Class ', df.iloc[index])
                
        print()


def main():

    #k = int(argv[1])

    sseList = []

    # read from data file
    # puts it into a data frame
    x_input = readFile()

    # note needed
    originalDF = x_input.copy()

    # drops the class attribute
    x_input = x_input.drop(columns=['class'])

    # the following are command line arguments

    # normal kmeans
    if 'norm' in argv:
        normal(x_input, False)
    
    elif 'codeCheck' in argv:
        normal(x_input, True)

    # senesitivity kmeans
    elif 'sens' in argv:
        print(sens(x_input))
    
    # kmeans++
    elif '++' in argv:
        print(kPlus(x_input))
    
    # top data points
    elif 'top' in argv:
        topThree(x_input, originalDF)
    
    # all options
    elif 'all' in argv:
        normal(x_input, False)
        senMean = sens(x_input)
        plusMean = kPlus(x_input)
        topThree(x_input, originalDF)
        print('sse result for sensitivity w/ kmeans')
        print(senMean)
        print()
        print('sse result for sensitivity w/ kmeans++')
        print(plusMean)
    
    else:
        print()
        print('read the README')



main()
