##### Clean up and load libraries
rm(list=ls())
setwd("./")
library(ggplot2) # So we can make graphcs
library(pROC) # Includes function to calculate the AUC metric
library(neuralnet)

# Read documentation for a neuralnet here:
# https://datascienceplus.com/fitting-neural-network-in-r/
# https://en.wikipedia.org/wiki/Artificial_neural_network
# https://cran.r-project.org/web/packages/neuralnet/neuralnet.pdf

##### Load Data
data(iris)

##### Data Cleaning, Center & Scale
# Note: the iris dataset is pristine, but in the real world you need to do this
# Note: Make sure you treat your test and train datasets the same if they come from different files!
print("Number of NA's: ")
print(sum(is.na(iris))) # Check for missing data
print(str(iris)) # Look at data structure
print(summary(iris)) # Look at data
# We are going to do our feature engineering earlier, and then center and scale our data
# Feature engineering
iris$Leaf.size <- iris$Sepal.Length * iris$Sepal.Width
# Split x/y datasets
iris_y <- iris$Species == "versicolor"
iris$Species <- NULL
# Center and scale
iris_scaled <- scale(iris, center = TRUE, scale = TRUE)
print(summary(iris_scaled))
# Notice how the column means = 0 and the standard deviation = 1

##### Split test and train datasets 60/40
ix <- sample(nrow(iris_scaled), round(0.6 * nrow(iris_scaled))) # Create index of rows
train <- iris_scaled[ix, ] # We'll use this to train our model
test <- iris_scaled[-ix, ] # We'll use this to test our model
train_y <- iris_y[ix]
test_y <- iris_y[-ix]

# Note: In this example we use a test and training set (and we know the test response value)
# You might consider using a training, validation, and test set (since you won't know the test reponse value)
# You would use val in place of test, and only use test for the final submission

##### Check it over
print(head(test))
print(head(train))

##### Neural Classification
# Documentation: ?neuralnet
# Neuralnet takes one input dataframe
train_input <- data.frame(train, y = as.numeric(train_y))
print(head(train_input))
nn <- neuralnet(y ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width + Leaf.size, 
                data = train_input, # Specify the input data
                hidden = c(8, 4), # Define the size of the neural network
                threshold = 1e-4, # Define stopping criteria
                rep = 1000, # Maximum number of training iterations
                act.fct = "logistic", # Tell it to train a classifier
                linear.output = FALSE # Necessary for a classifier
)
print(summary(nn))

# Let's take a look at our neural network
# NOTE!!! On my computer, I think there's a bug where this went on forever, so you might have to press the red "STOP" bottom
# In the top-left of the console. 
# You've been warned!
plot(nn)

##### Assess model performance
# Created predicted results for the test dataset
test_predicted <- predict(nn, newdata = test, type = "prob")

# Combine variable for ggplot
output <- data.frame(Actual = test_y, Predicted = test_predicted) # Make sure we get the right column
# Plot density on scatterplot
ggplot(output, aes(x = Predicted, y = Actual)) + geom_point()
# Plot double histogram
ggplot(output, aes(x = Predicted, fill = Actual)) + geom_histogram(alpha = 0.5, position = "identity", bins = 10)
# Calculate AUC metric
print(auc(output$Actual, output$Predicted)) 
# Note: The order of Y_actual, Y_Predicted is important
# An AUC of 0.5 is as good as random and an AUC of 1.0 is perfect

##### Save results
# We need to save our results to a file for submission
results <- data.frame(ID = 1:nrow(output), p = output$Predicted)
# Note: I'm creating an ID column here, but if you have an ID column, you would use it
write.csv(results, file = "neuralnet_results.csv", row.names = FALSE)


