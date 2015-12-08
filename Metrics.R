################################################################################

TruePostive <- function(labels, predictions, currentLabel) {
  tp <- sum((labels == currentLabel) * (predictions == currentLabel))
  return (tp)
}

TrueNegative <- function(labels, predictions, currentLabel) {
  tn <- sum((labels != currentLabel) * (predictions != currentLabel))
  return (tn)
}

FalsePostive <- function(labels, predictions, currentLabel) {
  fp <- sum((labels != currentLabel) * (predictions == currentLabel))
  return (fp)
}

FalseNegative <- function(labels, predictions, currentLabel) {
  fn <- sum((labels == currentLabel) * (predictions != currentLabel))
  return (fn)
}

################################################################################

CalculateAllMetrics <- function(numLabels, trainLabels, predictedLabels) {
  recalls <- rep(0.0, 10)
  precisions <- rep(0.0, 10)
  specificities <- rep(0.0, 10)
  fMeasures <- rep(0.0, 10)
  falseDiscoveryRates <- rep(0.0, 10)
  for (label in 1:numLabels) {
    truePositive <- TruePostive(trainLabels, predictedLabels, (label - 1))
    trueNegative <- TrueNegative(trainLabels, predictedLabels, (label - 1))
    falsePositive <- FalsePostive(trainLabels, predictedLabels, (label - 1))
    falseNegative <- FalseNegative(trainLabels, predictedLabels, (label - 1))
    
    recalls[label] <- truePositive / (truePositive + falseNegative)
    precisions[label] <- truePositive / (truePositive + falsePositive)
    specificities[label] <- trueNegative / (trueNegative + falsePositive)
    falseDiscoveryRates[label] <- falsePositive / (falsePositive + truePositive)
    
    fMeasures[label] <- 2 * (recalls[label] * precisions[label]) /
                            (recalls[label] + precisions[label])
  }
  metrics <- list("recalls" = recalls, "precisions" = precisions,
                  "specificities" = specificities, "fMeasures" = fMeasures,
                  "falseDiscoveryRates" = falseDiscoveryRates)
  return (metrics)
}

PlotROC <- function(currentLabel, labels, predictedProbabilities) {
  pred <- prediction(predictedProbabilities, labels)
  perf <- performance(pred, "tpr", "fpr")
  par(mar=c(5, 5, 2, 2), xaxs = "i", yaxs = "i", cex.axis=1.3, cex.lab=1.4)
  plot(perf, col="blue", lty=3, lwd=3)
}

PrintMetrics <- function(recalls, precisions, specificities, fMeasures,
                         falseDiscoveryRates) {
  print(paste(c("Recalls:", recalls), collapse=" "))
  print("_________________")
  print(paste(c("Precisions:", precisions), collapse=" "))
  print("_________________")
  print(paste(c("Specifities:", specificities), collapse=" "))
  print("_________________")
  print(paste(c("F-Measures:", fMeasures), collapse=" "))
  print("_________________")
  print(paste(c("FDR:", falseDiscoveryRates), collapse=" "))
}

################################################################################
