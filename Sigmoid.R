Sigmoid <- function(x) {
  # Computes sigmoid function on elements of the matrix x
  # Args:
  #   x: matrix
  #
  # Returns:
  #   A matrix of sigmoid function values on the elements of the matrix x
  return (1.0 / (1.0 + exp(-x)))
}
