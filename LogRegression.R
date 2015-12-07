source("Sigmoid.R")

################################################################################

LogRegressionCost <- function(theta, x, y, regularizationRate=0.0) {
  # Computes cost and gradient for logistic regression with regularization
  # Args:
  #   theta: vector of parameters
  #   x: matrix of features
  #   y: vector of labels
  #   regularizationRate: regularization rate
  #
  # Returns:
  #   cost and gradient of using theta as the parameter for regularized
  #   logistic regression
  sigmoid <- Sigmoid(x %*% theta)

  kEpsilon <- 0.1e-15
  error <- -t(1 - y) %*% log(1 - sigmoid + kEpsilon)
  error <- error - t(y) %*% log(sigmoid + kEpsilon)
  error <- error + regularizationRate * (t(theta) %*% theta - theta[1]^2) / 2.0
  error <- error / nrow(x)
  
  tmpTheta <- theta
  tmpTheta[1] <- 0
  gradient <- (t(x) %*% (sigmoid - y) + regularizationRate * tmpTheta)
  gradient <- gradient / nrow(x)
  
  result <- list("error" = error, "gradient" = gradient)
  return (result)
}

################################################################################

LogRegressionTrain <- function(x, y, stochastic=FALSE, batchSize=300,
                               regularizationRate=0.0, learningRate=1.0,
                               maxNumIters=10000, verbouse=TRUE) {
  # Implementation of logistic regression train algorithm using gradient descent
  # Args:
  #   x: matrix of features
  #   y: vector of labels
  #   stochastic: use stochastic gradient descent
  #   batchSize: size of train batch for stochastic gradient descent
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
  if (!stochastic) {
    batchSize <- nrow(x)
  }
  indices <- sample(1:nrow(x), batchSize)
  previousCostResult <- LogRegressionCost(theta, x[indices,],
                                          as.matrix(y[indices]),
                                          regularizationRate)
  costResult <- previousCostResult
  for(iter in 1:maxNumIters) {
    step <- 0.8 * previousCostResult$gradient + 0.2 * costResult$gradient
    theta <- theta - learningRate * step
    indices <- sample(1:nrow(x), batchSize)
    costResult <- LogRegressionCost(theta, x[indices,], as.matrix(y[indices]),
                                    regularizationRate)
    if (abs(costResult$error - previousCostResult$error) < kThreshold) {
      break
    }
    if (previousCostResult$error - costResult$error < 0) {
      learningRate <- learningRate / 1.5
    }
    if (verbouse && iter %% 1000 == 0) {
      message <- paste(c("Current error:", costResult$error), collapse=" ")
      write(message, stderr())
    }
    previousCostResult <- costResult
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
  #   threshold: threshold for probability of getting "1" label
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

OneVsAllLogRegressionTrain <- function(x, y, numLabels, stochastic=FALSE,
                                       batchSize=300, regularizationRate=0.0,
                                       learningRate=1.0, maxNumIters=10000,
                                       verbouse=TRUE) {
  # Trains multiple logistic regression classifiers and returns numLabels
  # classifiers in a matrix allTheta, where the i-th row of allTheta 
  # corresponds to the classifier for label i
  # Args:
  #   x: matrix of features
  #   y: vector of labels
  #   stochastic: to use stochastic gradient descent
  #   batchSize: size of train batch for stochastic gradient descent
  #   numLabels: number of different labels
  #   regularizationRate: regularization rate
  #   learningRate: learning rate
  #   maxNumIters: maximal numbers of gradient descent iterations
  #
  # Returns:
  #   matrix of parameters theta for every classifier
  allTheta <- matrix(0.0, numLabels, ncol(x) + 1)
  for (lableCounter in 1:numLabels) {
    message <- paste(c("Training classifier number", lableCounter),
                    collapse=" ")
    write(message, stderr())
    allTheta[lableCounter,] <- LogRegressionTrain(x, (y == lableCounter),
                                                  stochastic, batchSize,
                                                  regularizationRate,
                                                  learningRate, maxNumIters,
                                                  verbouse)
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
