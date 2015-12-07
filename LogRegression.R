source("Sigmoid.R")

################################################################################

LogRegressionCost <- function(theta, x, y, regularizationRate=0.0) {
  # Computes cost for logistic regression with regularization
  # Args:
  #   theta: vector of parameters
  #   x: matrix of features
  #   y: vector of labels
  #   regularizationRate: regularization rate
  #
  # Returns:
  #   the cost of using theta as the parameter for regularized
  #   logistic regression
  kEpsilon <- 0.1e-15
  sigmoid <- Sigmoid(x %*% theta)
  error <- -t(1 - y) %*% log(1 - sigmoid + kEpsilon)
  error <- error - t(y) %*% log(sigmoid + kEpsilon)
  error <- error + regularizationRate * (t(theta) %*% theta - theta[1]^2) / 2.0
  error <- error / nrow(x)
  return (error)
}

LogRegressionGradient <- function(theta, x, y, regularizationRate=0.0) {
  # Computes gradient for logistic regression with regularization
  # Args:
  #   theta: vector of parameters
  #   x: matrix of features
  #   y: vector of labels
  #   regularizationRate: regularization rate
  #
  # Returns:
  #   gradient of the cost of using theta as the parameter for regularized
  #   logistic regression
  sigmoid <- Sigmoid(x %*% theta)
  tmpTheta <- theta
  tmpTheta[1] <- 0
  gradient <- (t(x) %*% (sigmoid - y) + regularizationRate * tmpTheta)
  gradient <- gradient / nrow(x)
  return (gradient)
}

################################################################################

LogRegressionTrain <- function(x, y, regularizationRate=0.0, learningRate=0.5,
                               maxNumIters=10000) {
  # Implementation of logistic regression train algorithm using gradient descent
  # Args:
  #   x: matrix of features
  #   y: vector of labels
  #   regularizationRate: regularization rate
  #   learningRate: learning rate
  #   maxNumIters: maximal numbers of gradient descent iterations
  #
  # Returns:
  #   vector of parameters theta
  kThreshold <- 0.1e-10
  bias <- rep(1, nrow(x))
  x <- cbind(bias, x)
  theta <- matrix(0.0, ncol(x), 1)
  previousError <- LogRegressionCost(theta, x, y, regularizationRate)
  for(iter in 1:maxNumIters) {
    
    theta <- theta - learningRate * LogRegressionGradient(theta, x, y,
                                                          regularizationRate)
    error <- LogRegressionCost(theta, x, y, regularizationRate)
    if (abs(error - previousError) < kThreshold) {
      break
    }
    if (iter %% 1000 == 0) {
      write(error, stderr())
    }
    previousError <- error
  }
  return(theta)
}

LogRegressionStochasticTrain <- function(x, y, batchSize=300,
                                         regularizationRate=0.0,
                                         learningRate=0.5,
                                         maxNumIters=10000) {
  # Implementation of logistic regression train algorithm using
  # stochastic gradient descent
  # Args:
  #   x: matrix of features
  #   y: vector of labels
  #   regularizationRate: regularization rate
  #   learningRate: learning rate
  #   maxNumIters: maximal numbers of gradient descent iterations
  #
  # Returns:
  #   vector of parameters theta
  kThreshold <- 0.1e-10
  bias <- rep(1, nrow(x))
  x <- cbind(bias, x)
  theta <- matrix(0.0, ncol(x), 1)
  previousError <- LogRegressionCost(theta, x, y, regularizationRate)
  for(iter in 1:maxNumIters) {
    batchIndices = sample(1:nrow(x), batchSize)
    xBatch = x[batchIndices]
    yBatch = y[batchIndices]
    theta <- theta - learningRate * LogRegressionGradient(theta, xBatch, yBatch,
                                                          regularizationRate)
    error <- LogRegressionCost(theta, xBatch, yBatch, regularizationRate)
    if (abs(error - previousError) < kThreshold) {
      break
    }
    if (iter %% 1000 == 0) {
      write(error, stderr())
    }
    previousError <- error
  }
  return(theta)
}

################################################################################

LogRegressionPredictProba <- function(theta, x) {
  # Predicts probability of "1" label, using vector of parameters theta
  # Args:
  #   theta: vector of parameters
  #   x: matrix of features
  #
  # Returns:
  #   vector of probabilities
  bias <- rep(1, nrow(x))
  x <- cbind(bias, x)
  sigmoid <- Sigmoid(x %*% theta)
  return (sigmoid)
}

LogRegressionPredict <- function(theta, x, threshold=0.5) {
  # Predict label, using vector of parameters theta
  # Args:
  #   theta: vector of parameters
  #   x: matrix of features
  #   threshold: threshold for probability to have "1" label
  #
  # Returns:
  #   vector of labels
  probabilities <- LogRegressionPredictProba(theta, x)
  labels <- rep(0, nrow(x))
  for (objCounter in 1:nrow(x)) {
    if (probabilities[objCounter] > threshold) {
      labels[objCounter] <- 1
    }
  }
  return (labels)
}

################################################################################

OneVsAllLogRegression <- function(x, y, numLabels, regularizationRate=0.0, 
                                  learningRate=1.0, maxNumIters=10000) {
  # Trains multiple logistic regression classifiers and returns numLabels
  # classifiers in a matrix allTheta, where the i-th row of allTheta 
  # corresponds to the classifier for label i
  # Args:
  #   x: matrix of features
  #   y: vector of labels
  #   numLabels: number of different labels
  #   regularizationRate: regularization rate
  #   learningRate: learning rate
  #   maxNumIters: maximal numbers of gradient descent iterations
  #
  # Returns:
  #   matrix of parameters theta for every classifier
  allTheta <- matrix(0.0, numLabels, ncol(x) + 1)
  for (lableCounter in 1:numLabels) {
    message = paste(c("Training classifier number", lableCounter), collapse = " ")
    write(message, stderr())
    allTheta[lableCounter,] <- LogRegressionTrain(x, (y == lableCounter), 
                                                  regularizationRate,
                                                  learningRate, maxNumIters)
  }
  return (allTheta)
}

MultiLabelPredictProba <- function(allTheta, x) {
  # Predicts probability of every label, using marix of parameters allTheta
  # Args:
  #   allTheta: matrix of parameters theta for every classifier
  #   x: matrix of features
  #
  # Returns:
  #   matrix of probabilities
  bias <- rep(1, nrow(x))
  x <- cbind(bias, x)
  probabilities <- Sigmoid(allTheta %*% t(x))
  return (probabilities)
}

MultiLabelLogRegressionPredict <- function(allTheta, x) {
  # Predicts labels, using marix of parameters allTheta
  # Args:
  #   allTheta: matrix of parameters theta for every classifier
  #   x: matrix of features
  #
  # Returns:
  #   vector of labels
  probabilities <- MultiLabelPredictProba(allTheta, x)
  labels <- max.col(t(probabilities))
  return (labels)
}

################################################################################
