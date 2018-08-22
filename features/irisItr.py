import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import seaborn as sns 
import pandas as pd
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import minkowski


# Flower class
class Flower:
    def __init__(self):
        self.name = ''

        # Attributes
        self.sepalLength = []
        self.sepalWidth = []
        self.petalLength = []
        self.petalWidth = []

        self.binNumArr = []
        self.theBin = []

        # Bin labels
        self.binSLlabel = []
        self.binSWlabel = []
        self.binPLlabel = []
        self.binPWlabel = []

        # Bins
        self.binSL = []
        self.binSW = []
        self.binPL = []
        self.binPW = []

# Wine class
class Wine:
    def __init__(self):
        self.name = ''

        # Attributes
        self.alcohol = []
        self.malicAcid = []
        self.ash = []
        self.alcalinity = []
        self.magnesium = []
        self.totalPhenols = []
        self.flavanoids = []
        self.nonflavanoids = []
        self.proanthocyanins = []
        self.colorIntensity = []
        self.hue = []
        self.dilution = []
        self.proline = []

        # Bin Labels
        self.alcLabel = []
        self.malLabel = []
        self.ashLabel = []

        # Bins
        self.alcBin = []
        self.malBin = []
        self.ashBin = []
 



# A majority of the functions will have the same
# names for the arguments
# fo - class objects
# fa - class attributes
# fl - class labels
# fb - class bins

# Code provided by the TA
def binWork(fo, fa, fl, fb):

    # Bin numbers
    binArr = [5, 10, 50, 100]

    for flower in fo:

        #print("\t !!!" + flower.name + "!!!")

        for attr, binVar, label in zip(fa, fb, fl):
            #print("Working on:" + attr)

            b = getattr(flower, attr)[0]

            '''
            if (len(argv) == 3):
                if (argv[2] == "sort"):
                    b.sort()
            '''

            min_b = min(b)
            max_b = max(b)

            # [ 5 10 50 100 ] inside binArr
            for numOfbins, cnt in zip(binArr, range(4)):
                # print("numOfbins:", numOfbins)

                binRange = float((max_b - min_b) / numOfbins)

                r1 = min_b
                r2 = min_b + binRange

                getattr(flower, binVar).append([])
                getattr(flower, label).append([])

                # Iterates through bin number ex 0 - 5, 0 - 10, 0 - 50
                # 0 - 100
                for binNum in range(numOfbins):
                    c = b[b >= r1]

                    if (binNum == numOfbins - 1):
                        lb = c[c <= max_b]
                    else:
                        lb = c[c < r2]

                    if (binNum == numOfbins - 1):
                        string = str(r1)[:7] + "-" + str(max_b)[:7]
                    else:
                        string = str(r1)[:7] + " - " + str(r2)[:7]


                    getattr(flower, binVar)[cnt].append(lb)
                    getattr(flower, label)[cnt].append(string)

                    r1 = r2
                    r2 = r2 + binRange


# Creates histograms
# fo - class objects
# fa - class attributes
# fb - class bins
# fl - class labels
# arg - command line arguments
def histo(fo, fa, fb, fl, arg):
    
    if (arg == 'flower'):
        fileDir = ['setosa/is_', 'versicolour/iv_', 'virginica/ivg_']
        flowerName = ['Iris Setosa ', 'Iris Versicolour ', 'Iris Virginica ']
        nameOfAttr = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        colorArr = ['skyblue', 'darkred', 'darkseagreen', 'orange'] 

    elif (arg == 'wine'):
        fileDir = ['one/one_', 'two/two_', 'three/three_']
        flowerName = ['One ', 'Two ', 'Three ']
        nameOfAttr = ['Alcohol ', 'Malic Acid ', 'Ash ']
        attrArr = ['alc_', 'ma_', 'ash_']
        colorArr = ['skyblue', 'darkred', 'darkseagreen']


    # Iterates through the contents of the object
    # In order to create the histogram
    for flower, name, dir in zip(fo, flowerName, fileDir):
        #print(name)
        for bins, label, binCnt in zip(fb, fl, range(4)):
            for tmpCnt in range(4):
                allBins = getattr(flower, bins)[tmpCnt]
                #print("\t", bins)

                tmpLabel = getattr(flower, label)[tmpCnt]

                numOfbins = len(allBins)
                stringBin = str(numOfbins)
                freqArr = np.zeros(numOfbins)
                inds = np.arange(numOfbins)


                for i in range(numOfbins):
                    freqArr[i] = len(allBins[i])

                if numOfbins == 5:
                    width = 0.35
                    
                elif numOfbins == 10:
                    width = 0.35
                
                elif numOfbins == 50:
                    width = 0.35
                    plt.figure(figsize=(20, 5))
                
                elif numOfbins == 100:
                    width = 0.35
                    plt.figure(figsize=(20, 5))

                # Plotting
                p1 = plt.bar(inds, freqArr, width, color=colorArr[binCnt])
                plt.xlabel('Bins')
                plt.ylabel('Frequency')
                plt.title(name + nameOfAttr[binCnt] +  " Bin # " + stringBin)
                plt.xticks(inds, tmpLabel, rotation='vertical')
                plt.savefig(dir + nameOfAttr[binCnt] + stringBin, bbox_inches='tight')
                plt.close()
            

# Box plot creation
# fo - class objects
# fa - class attributes
# fa - class names
# arg - command line arguments
def boxPlot(fo, fa, fn, arg):

    if (arg == 'flower'):
        colors = ['skyblue', 'darkred', 'darkseagreen']
        nameOfAttr = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        attrArr = ['sl', 'sw', 'pl', 'pw']
        fileDir = ['setosa/bis_', 'versicolour/biv_', 'virginica/bivg_']
    
    elif (arg == 'wine'):
        colors = ['skyblue', 'darkred', 'darkseagreen']
        nameOfAttr = ['Alcohol ', 'Malic Acid ', 'Ash ']
        attrArr = ['alc', 'ma', 'ash']
        fileDir = ['one/bone_', 'two/btwo_', 'three/bthree_']

    # Iterates through the contents of the object to 
    # create the boxplot
    for flower, name, clr, idir in zip(fo, fn, colors, fileDir):
        for attr, atname, atTag in zip(fa, nameOfAttr, attrArr):
            temp = getattr(flower, attr)[0]

            temp2 = plt.boxplot(temp, patch_artist=True)
            #temp2.['boxes'].set_facecolor(clr)
            #plt.setp(color=clr)
            plt.title(name + " " + atname)
            plt.ylabel(atname)
            plt.savefig(idir + atTag)
            plt.close()

# Calculates the mean of the data
def mean(data):

    return sum(data) / (len(data) )


# Calculates the variance of the data
def variance(data):
    # Sum ( datai - DATA)^2 / n

    # Get the mean of the data
    meanVal = mean(data)

    listVal = []

    # Subtract each value in data with the mean
    # And squres it and appens the result to the answer
    for ele in data:
        element = ele - meanVal
        listVal.append(element**2)

    # Summation divided by the number of elements
    # To get variance
    varianceVal = sum(listVal) / (len(data) )

    return varianceVal


# cov(X,Y)
def cov(X, Y):

    xList = []
    yList = []
    xyList = []

    # Mean calculation
    meanX = mean(X)
    meanY = mean(Y)

    # Calculation for (Xi --- X)
    for ele in X:
        xElement = ele - meanX
        xList.append(xElement)
    

    # Calculations for (Yi --- Y)
    for ele in Y:
        yElement = ele - meanY
        yList.append(yElement)
        
    # Production value of (Xi --- X)(Yi --- Y)
    xyList = []
    for xEle, yEle in zip(xList, yList):
        xyElement = xEle * yEle
        xyList.append(xyElement)
    
    # Summation of (Xi --- X)(Yi --- Y)
    xySum = sum(xyList)

    # Divide by the numerator by the denominator
    covVal = xySum / (len(X))

    return covVal


# Peasrson's correlation coefficient
def correlation( X, Y):
    
    # Get Cov
    covVal = cov( X, Y)

    # Get Variance
    varX = variance(X)
    varY = variance(Y)

    # Square roots the variance
    sdX = math.sqrt(varX)
    sdY = math.sqrt(varY)

    # Cov / the product of the variance
    corrVal = covVal / (sdX * sdY)

    #print("correlation value: ", corrVal )
    #corrVal = round(corrVal)

    # Returns the coefficient
    #print(corrVal)
    return abs(corrVal)

# Combines attribute data together regardless
# of class
def combineData(fo, fa):

    # Temp List to hold a list
    tempArr = []

    # Used to combine multiple list into one
    combArr = []

    # List of list that holds attributes of all 3 classes
    finalArr = []

    # Gathering the list from the seperated list
    for attr in fa:
        for flower in fo:
            temp = getattr(flower, attr)[0]
            tempArr.append(temp)

        # Combining the list
        for i in range(3):
            combArr.extend(tempArr[i])
        
        finalArr.append(combArr)

        # Clearing the list
        tempArr = []
        combArr = []

    return finalArr


# Creates the heat map for correlation data
# matrix - matrix
# arg - command line argument
def heatMap(matrix, arg):
    sns.set(font_scale=2)

    if arg == 'flower':
        labelList = ['Sepal\nLenght', 'Sepal\nWidth', 'Petal\nLength', 'Petal\nWidth']
        titleName = 'Flower '
        fileName = 'heatmapFlower'
        rotationVal = 'horizontal'
    
    elif arg == 'wine':
        labelList = ['alcohol', 'malic\n acid', 'ash', 'alcalinity', 'magnesium',
                        'magnesium', 'totalPhenols', 'flavanoids', 'nonflavanoids',
                        'proanthocyanins', 'colorIntensity', 'hue', 'dilution', 'proline' ]
        titleName = 'Wine '
        fileName = 'heatmapWine'
        rotationVal = 'vertical'

    # Plotting
    plt.figure(figsize=(30, 20))
    ax = sns.heatmap(matrix, annot=True, cmap="YlGnBu", xticklabels=True, yticklabels=True)
    ax.set_yticklabels(labelList, rotation=0)
    ax.set_xticklabels(labelList, rotation=rotationVal)
    ax.set_title(titleName + 'Correalation Matrix')


    plt.savefig(fileName, bbox_inches='tight')
    plt.close()


# heatmap for distance
# matrix - matrix 
# p - p value
# arg - command line argument
def heatMap2(matrix, p, arg):
    sns.set(font_scale=4)

    labelList = []

    if arg == 'flower':
        for i in range(len(matrix[0])):
            tempString = 'Flower ' + str(i + 1)
            labelList.append(tempString)


        titleName = 'Flower '
        fileName = 'heatmaps/heatmapFlowerDistanceP_' + str(p)
        rotationVal = 'vertical'
    
    elif arg == 'wine':
        for i in range(len(matrix[0])):
            tempString = 'Wine ' + str(i + 1)
            labelList.append(tempString)

        titleName = 'Wine '
        fileName = 'heatmaps/heatmapWineDistanceP_' + str(p)
        rotationVal = 'vertical'

    # Plotting
    plt.figure(figsize=(250, 250))
    ax = sns.heatmap(matrix, cmap="YlGnBu", xticklabels=True, yticklabels=True)
    ax.set_yticklabels(labelList, rotation=0)
    ax.set_xticklabels(labelList, rotation=rotationVal)
    ax.figure.axes[-1].yaxis.label.set_size(10)
    ax.set_title(titleName + 'Distance Matrix: P - ' + str(p), fontsize=40)


    plt.savefig(fileName, bbox_inches='tight')
    plt.close()


# creation of matrix
# allAttr - all attributes
# arg - command line arguments
def createMatrix(allAttr, arg):

    # Determine matrix size
    nMatrixSize = len(allAttr)

    # matrix[X][Y]
    matrix = np.zeros((nMatrixSize ,nMatrixSize))

    # Fills the matrix with the values of pearsons
    for xAxis, xPos in zip(allAttr, range(nMatrixSize)):
        for yAxis, yPos in zip(allAttr, range(nMatrixSize)):
            matrix[xPos][yPos] = correlation(xAxis, yAxis)
            #print('Correct value: ', pearsonr(xAxis, yAxis), "\n")
    
    # Zero out everything on the lower triangle
    # of the matrix
    matrix = np.triu(matrix)

    if arg == 'flower':
        fileDir = 'matrix/flower_correlation_matrix.txt'

    elif arg == 'wine':
        fileDir = 'matrix/wine_correlation_matrix.txt'

    myFile = open(fileDir, 'w')

    # Writing matrix to a file
    for itr in matrix:
        for temp in itr:

            if len(str(temp)) != 7:
                newTemp = '{0:07f}'.format(temp)

            myFile.write(str(newTemp)[:7] + '\t')
        
        myFile.write('\n')

    
    myFile.close()

    heatMap(matrix, arg)


# Creates scatter plot
# temp - holds all values from the data set
def scatter(temp):

    # name  labels 
    attrArr = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    # location directory
    locDir = 'scatter/'

    # Will hold class names
    myList = []

    # Filling a array with the class names
    for i in range(150):
        if i < 50:
            myList.append('Setosa')
        elif i < 100:
            myList.append('Versicolour')
        elif i < 150:
            myList.append('Virginica')

    # creating a panda data frame (2D)
    # With columns being the attributes and class
    # With their respective values under the columns
    myData = pd.DataFrame({'Sepal Length': temp[0], 'Sepal Width': temp[1], 'Petal Length': temp[2], 
                           'Petal Width': temp[3], 'Class': myList})    


    # Creates the scatter plots - 16 will be made
    for myX in attrArr:
        for myY in attrArr:
            ax = sns.lmplot(x=myX, y=myY, data=myData, fit_reg=False, hue='Class', palette='husl', markers=['<', 'v', '>'] )
            fig = ax.fig
            fig.suptitle(myX + " vs " + myY)
            plt.savefig(locDir + myX + " vs " + myY)
            plt.close()
    


# Gets the nearest data point
# matrix - matrix 
# p - p value
# arg - command line arguments
def getNearest(matrix, p, arg):

    if arg == 'flower':
        limits = [50, 100, 150]
        fileDir = 'neighbors/flower_neighbor_' + str(p) + '.txt'
        labels = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

    elif arg == 'wine':
        labels = ['1', '2', 3]
        limits = [59, 130, 178]
        fileDir = 'neighbors/wine_neighbor_' + str(p) + '.txt'


    # Gets the length of the matrix
    axisLength = len(matrix)

    # csv matrix for debugging
    csvMatrix = np.around(matrix, 4)
    np.savetxt("checkME.csv", csvMatrix, delimiter=",")
    
    tempList = []
    neighborsList = []

    lowestValue = 9999
    colLocation = 0
    rowLocation = 0

    # Finds the lowest value within each row
    for row in range(axisLength):
        for col in range(axisLength):
            if col != row:

                curr = matrix[row][col]
                
                if curr < lowestValue:
                    lowestValue = curr
                    colLocation = col
                    rowLocation = row
            
        
        tempList = [lowestValue, rowLocation, colLocation]
        neighborsList.append(tempList)
        tempList = []
        lowestValue = 9999
    
    myFile = open(fileDir, 'w')

    # gets the class label
    for itr in neighborsList:
        temp = itr[2]
        #temp2 = itr[2]
    
        if temp < limits[0]:
            itr.append(labels[0])

        elif temp < limits[1]:
            itr.append(labels[1])

        elif temp < limits[2]:
            itr.append(labels[2])
        
        myFile.write(str(itr) + '\n')
        #print(itr)
    
    myFile.close()
    
    # The following get the percentage 
    # of how often a class is lowest neighbor relative 
    # to each other
    class1 = 0.0
    class2 = 0.0
    class3 = 0.0


    # Increments for each hit
    for itr in neighborsList:
        endCheck = itr[3]

        if endCheck == labels[0]:
            class1 += 1

        if endCheck == labels[1]:
            class2 += 1

        if endCheck == labels[2]:
            class3 += 1

    # Get the number of hits
    totalHits = class1 + class2 + class3
    print(totalHits)
    
    # Get the percentage
    class1 = class1 / totalHits
    class2 = class2 / totalHits
    class3 = class3 / totalHits

    print('P value of ', p)
    print(labels[0], ': ', class1 * 100, '%')
    print(labels[1], ': ', class2 * 100, '%')
    print(labels[2], ': ', class3 * 100, '%')

    print()
    #print(matrix)



# minkowskidistance
def iminkowski( X, Y, p):
    xyList = []

    # Calculation for (Xi --- Yi)
    for xElement, yElement in zip(X, Y):
        xyElement = abs(xElement - yElement)
        xyElement = pow(xyElement, p)
        xyList.append(xyElement)
    
    
    # Summation of (Xi --- Yi)
    xySum = sum(xyList)

    if p == 2:
        xySum = math.sqrt(xySum)

    return xySum

def kowski(allAttr, p, arg):
    flowerCell = []

    # Getting size of attributes 
    mSize = len(allAttr[0])

    matrix = np.zeros((mSize,mSize))

    if arg == 'flower':
        dataType = 'matrix/flower_'

    elif arg == 'wine':
        dataType = 'matrix/wine_'

    pType = str(p)

    matrixFileName = dataType + pType + '_matrix.txt'
    
    temp = []

    # Creating a element for arrays that holds
    # each iris / wine
    for row in range(mSize):
        for attr in range(len(allAttr)):
            temp.append([])
            temp[attr] = allAttr[attr][row]

        flowerCell.append(temp)            
        temp = []


    # calculating distance
    for xAxis, xPos in zip(flowerCell, range(mSize)):
        for yAxis, yPos in zip(flowerCell, range(mSize)):
            matrix[xPos][yPos] = iminkowski(xAxis, yAxis, p)
            #print(matrix[xPos][yPos], " --- ", minkowski(xAxis, yAxis, p))

    # Writes the matrix to a text file
    myFile = open(matrixFileName, 'w')

    nonZeroMatrix = matrix
    matrix = np.triu(matrix)

    # writing matrix to a text file
    for row in range(mSize):
        for col in range(mSize):
            temp = matrix[row][col] 

            if len(str(temp)) != 6:
                temp = '{0:06f}'.format(temp)

            myFile.write(str(temp)[:6] + "\t ")
        
        myFile.write('\n')

    myFile.close()

    heatMap2(matrix, p, arg)
    #print(matrix)
    getNearest(nonZeroMatrix, p, arg)

def main(argv):

    if argv[1] == 'flower':
        setosa = Flower()
        versicolour = Flower()
        virginica = Flower()

        sml = [0, 50, 100]
        big = [50, 100, 150]
        flowerObjs = [setosa, versicolour, virginica]
        flowerNames = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
        flowerAttr = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']
        flowerBins = ['binSL', 'binSW', 'binPL', 'binPW']
        flowerBinsLabel = ['binSLlabel', 'binSWlabel', 'binPLlabel', 'binPWlabel']
        a = np.loadtxt('iris.data', delimiter=',', usecols=(0, 1, 2, 3))

    elif argv[1] == 'wine':
        one = Wine()
        two = Wine()
        three = Wine()

        sml = [0, 59, 130]
        big = [59, 130, 178]
        flowerObjs = [one, two, three]
        flowerNames = ['One', 'Two', 'Three']
        flowerAttr = ['alcohol', 'malicAcid', 'ash', 'alcalinity', 'magnesium',
                        'totalPhenols', 'flavanoids', 'nonflavanoids', 'proanthocyanins',
                         'colorIntensity', 'hue', 'dilution', 'proline' ]
        flowerBins = ['alcBin', 'malBin', 'ashBin']
        flowerBinsLabel = ['alcLabel', 'malLabel', 'ashLabel']
        a = np.loadtxt('wine.data', delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                             10, 11, 12, 13))

    for nClass, nm, sm, bg in zip(flowerObjs, flowerNames, sml, big):
        nClass.name = nm

        #print(len(flowerAttr))

        for attr, cnt in zip(flowerAttr, range(len(flowerAttr))):
            getattr(nClass, attr).append(a[sm:bg, cnt])

    #checkValid(flowerObjs, flowerAttr, flowerBins, flowerBinsLabel)
    #printBins(flowerObjs, flowerBins, flowerBinsLabel)

    # Organizes the data differently
    # In the class objects the attributes are separated 
    # by its class
    # This function combines the attributes together regardless
    # of the class
    allFlowerAttr = combineData(flowerObjs, flowerAttr)
    #print(allFlowerAttr[0])

    if (len(argv) > 1):
        if argv[2] == 'bins':
            binWork(flowerObjs, flowerAttr, flowerBinsLabel, flowerBins)
            histo(flowerObjs, flowerAttr, flowerBins, flowerBinsLabel, argv[1])
            boxPlot(flowerObjs, flowerAttr, flowerNames, argv[1])

        elif argv[2] == 'hist':
            binWork(flowerObjs, flowerAttr, flowerBinsLabel, flowerBins)
            histo(flowerObjs, flowerAttr, flowerBins, flowerBinsLabel, argv[1])

        elif argv[2] == 'box':
            binWork(flowerObjs, flowerAttr, flowerBinsLabel, flowerBins)
            boxPlot(flowerObjs, flowerAttr, flowerNames, argv[1])
        
        elif argv[2] == 'corr':
            createMatrix(allFlowerAttr, argv[1])
        
        # Only for flower
        elif argv[2] == 'scatter' and argv[1] == 'flower':
            scatter(allFlowerAttr)

        elif argv[2] == 'distance':
            kowski(allFlowerAttr, 1, argv[1])
            kowski(allFlowerAttr, 2, argv[1])
        
        else:
            print('Please use bins or hist or box or corr scatter or distance as your third argument')
            print('Example: python3 irisItr.py flower bin')

    else:
        print('Please enter arguments')
        print('Example: python3 irisItr.py flower bin')
        print('Use either flower or wine as your second argument')
        print('Please use bins or hist or box or corr scatter or distance as your third argument')

main(sys.argv)