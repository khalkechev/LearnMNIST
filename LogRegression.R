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

  gradient <- (t(x) %*% (sigmoid - y) + regularizationRate * c(0, theta[-1]))
  gradient <- gradient / nrow(x)
  
  result <- list("error" = error, "gradient" = gradient)
  return (result)
}

################################################################################

LogRegressionTrain <- function(x, y, stochastic=FALSE, batchSize=300,
                               regularizationRate=0.0, learningRate=1.0,
                               maxNumIters=1000, momentum=0.0, verbose=TRUE) {
  # Implementation of logistic regression train algorithm using gradient descent
  # Args:
  #   x: matrix of features
  #   y: vector of labels
  #   stochastic: use stochastic gradient descent or not (default FALSE)
  #   batchSize: size of batch for stochastic gradient descent (default 300)
  #   regularizationRate: regularization rate (default 0.0)
  #   learningRate: learning rate (default 1.0)
  #   maxNumIters: maximal numbers of gradient descent iterations (default 1000)
  #   momentum: momentum technique coefficient (default 0.0)
  #   verbose: print additional details of training (default TRUE)
  #
  # Returns:
  #   vector of parameters theta
  kErrorsThreshold <- 0.1e-10
  bias <- rep(1, nrow(x))
  x <- cbind(bias, x)
  theta <- matrix(0.0, ncol(x), 1)
  if (!stochastic) {
    batchSize <- nrow(x)
  }
  indices <- sample(1:nrow(x), batchSize)
  previousCost <- LogRegressionCost(theta, x[indices,], as.matrix(y[indices]),
                                    regularizationRate)
  previousStep <- previousCost$gradient
  cost <- previousCost
  for(iter in 1:maxNumIters) {
    step <- - (momentum * previousStep + (1 - momentum) * cost$gradient)
    theta <- theta + learningRate * step
    indices <- sample(1:nrow(x), batchSize)
    cost <- LogRegressionCost(theta, x[indices,], as.matrix(y[indices]),
                              regularizationRate)
    if (abs(cost$error - previousCost$error) < kErrorsThreshold) {
      break
    }
    if (previousCost$error < cost$error) {
      learningRate <- learningRate / 1.5
    }
    if (verbose && iter %% 500 == 0) {
      message <- paste(c("Current error:", cost$error), collapse=" ")
      write(message, stderr())
    }
    previousCost <- cost
    previousStep <- step
  }
  return(theta)
}

MultiLabelLogRegressionTrain <- function(x, y, numLabels, stochastic=FALSE,
                                         batchSize=300, regularizationRate=0.0,
                                         learningRate=1.0, maxNumIters=1000,
                                         momentum=0.0, verbose=TRUE) {
  # Trains multiple logistic regression classifiers and returns numLabels
  # classifiers in a matrix allTheta, where the i-th row of allTheta
  # corresponds to the classifier for label i
  # Args:
  #   x: matrix of features
  #   y: vector of labels
  #   numLabels: number of different labels
  #   stochastic: to use stochastic gradient descent (default FALSE)
  #   batchSize: size of batch for stochastic gradient descent (default 300)
  #   regularizationRate: regularization rate (default 0.0)
  #   learningRate: learning rate (default 1.0)
  #   maxNumIters: maximal numbers of gradient descent iterations (default 1000)
  #   momentum: momentum technique coefficient (default 0.0)
  #   verbose: print additional details of training (default TRUE)
  #
  # Returns:
  #   matrix of parameters theta for every classifier
  allTheta <- matrix(0.0, numLabels, ncol(x) + 1)
  for (lableCounter in 1:numLabels) {
    if (verbose) {
      message <- paste(c("Training classifier number", lableCounter),
                       collapse=" ")
      write(message, stderr())
    }
    allTheta[lableCounter,] <- LogRegressionTrain(x, (y == (lableCounter - 1)),
                                                  stochastic, batchSize,
                                                  regularizationRate,
                                                  learningRate, maxNumIters,
                                                  momentum, verbose)
  }
  return (allTheta)
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
  probabilities <- Sigmoid(x %*% theta)
  return (probabilities)
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

MultiLabelLogRegressionPredictProba <- function(allTheta, x) {
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
  probabilities <- MultiLabelLogRegressionPredictProba(allTheta, x)
  labels <- max.col(t(probabilities)) - 1
  return (labels)
}

################################################################################
