source("LogRegression.R")

nFeatures <- 2
nData <- 1000

x <- matrix(runif(nData*(nFeatures), min = 0, max = 2), ncol = nData, nrow = nFeatures)
x = t(x)
dim(x)
k1 = 0.5
k2 = 1.5
b = 0
y <- matrix(0, nrow(x), 1)
for (i in 1:nrow(x)) {
  tan <- x[i,1] / x[i,2]
  if (tan > k2) {
    y[i] <- 2
  } else if (tan > k1) {
    y[i] <- 1
  }
}
plot(x[,1],x[,2],col=c("red","blue", "green")[y + 1])

train = x[1:500,]
y_train = y[1:500,]

test = x[501:1000,]
y_test = y[501:1000,]

plot(train[,1],train[,2],col=c("red","blue", "green")[y_train + 1])
plot(test[,1],test[,2],col=c("red","blue", "green")[y_test + 1])
plot(train[,1],train[,2],col=c("red","blue", "green")[y_train + 1])
plot(test[,1],test[,2],col=c("red","blue", "green")[y_test + 1])

allTheta <- MultiLabelLogRegressionTrain(train, y_train, 3,
                                         stochastic=TRUE,
                                         batchSize=300,
                                         regularizationRate=0.01,
                                         learningRate=1.0,
                                         maxNumIter=10000,
                                         momentum=0.0)

y_predicted = MultiLabelLogRegressionPredict(allTheta, train)
#calculate accuracy
print("accuracy on train data:")
print(sum(y_predicted == y_train)/length(y_train))

y_predicted = MultiLabelLogRegressionPredict(allTheta, test)
#calculate accuracy
print("accuracy on test data:")
print(sum(y_predicted == y_test)/length(y_test))
