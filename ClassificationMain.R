# This sciprt file contains a frame for learning handwritten digitals from the MNIST dataset
library(ROCR)
source("load_data.R")
source("LogRegression.R")
source("Metrics.R")

# load training data from files
data <- loadMNISTData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte")
trainLabels <- data$labels
trainData <- data$data

print(dim(trainData))
print(dim(trainLabels))
# trainingData should be 60000x784,  60000 data and 784 features (28x28), tha matrix trainData has 60000 rows and 784 columns
# trainingLabels should have 60000x1, one class label \in {0,1,...9} for each data.

#uncomment the following 3 lines to see the nth training example and its class label.
#n = 10;
#image(t(matrix(trainData[n, ], ncol=28, nrow=28)), Rowv=28, Colv=28, col = heat.colors(256),  margins=c(5,10))
#print("Class label:"); print(trainLabels[n])

# train a model
classifier <- MultiLabelLogRegressionTrain(trainData, trainLabels, numLabels=10,
                                           stochastic=TRUE, batchSize=1000,
                                           regularizationRate=0.01, learningRate = 1.0,
                                           maxNumIters=10000, momentum=0.1)

predictedLabels <- MultiLabelLogRegressionPredict(classifier, trainData)

#calculate accuracy on training data
print("accuracy on training data:")
print(sum(predictedLabels == trainLabels) / length(trainLabels))

#calculate the following error metric for each class obtained on the train data:
#Recall, precision, specificity, F-measure, FDR and ROC for each class separately. Use a package for ROC.

metrics <- CalculateAllMetrics(10, trainLabels, predictedLabels)
print("Class 0, Class 1, Class 2, Class 3, Class 4, Class 5, Class 6, Class 7, Class 8, Class 9")
PrintMetrics(metrics$recalls,
             metrics$precisions,
             metrics$specificities,
             metrics$fMeasures,
             metrics$falseDiscoveryRates)

predictedProbabilities <- MultiLabelLogRegressionPredictProba(classifier, trainData)

for (label in 0:9){
  PlotROC(label, (trainLabels == label), predictedProbabilities[label + 1, ])
}

# test the model
data <- loadMNISTData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte")
testLabels <- data$labels
testData <- data$data

print(dim(testData))
print(dim(testLabels))
#trainingData should be 10000x784,  10000 data and 784 features (28x28), tha matrix trainData has 10000 rows and 784 columns
#trainingLabels should have 10000x1, one class label \in {0,1,...9} for each data.

predictedLabels <- MultiLabelLogRegressionPredict(classifier, testData)

#calculate accuracy
print("accuracy on test data:")
print(sum(predictedLabels == testLabels) / length(testLabels))

#calculate the following error metric for each class obtained on the test data:
#Recall, precision, specificity, F-measure, FDR and ROC for each class separately. Use a package for ROC.
testMetrics <- CalculateAllMetrics(10, trainLabels, predictedLabels)
print("Class 0, Class 1, Class 2, Class 3, Class 4, Class 5, Class 6, Class 7, Class 8, Class 9")
PrintMetrics(testMetrics$recalls,
             testMetrics$precisions,
             testMetrics$specificities,
             testMetrics$fMeasures,
             testMetrics$falseDiscoveryRates)

predictedProbabilities <- MultiLabelLogRegressionPredictProba(classifier, testData)

for (label in 0:9){
  PlotROC(label, (trainLabels == label), predictedProbabilities[label + 1, ])
}
