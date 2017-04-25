# STAT 577 Final Project Experimental Script
# 2017-04-20

# Load Libraries
library(sqldf)
library(MASS)
library(class)
library(MASS)
library(tree)
require(randomForest)
library(gbm)

# Load Data
#DF = read.csv("data/DF.csv")
DF = read.csv("DF_wo_outliers.csv")
DF$household_key=NULL
DF$X=NULL
summary(DF)
head(DF)
dim(DF)
head(DF)

# Start Analysis 
# See homework 4 for best place to start.
n = 50
errmat = matrix(0,n,4)
set.seed(2)

for (i in 1:n){
    # Split data into Training and Test sample
  trainIndex <- sample(1:nrow(DF),400)
  test <- DF[-trainIndex,]
  train <- DF[trainIndex,]
    
  # Bagging
  BagMod <- randomForest(TotalValue~.,data=train,mtry=8)
  pred <- predict(BagMod,type = "response")
  BagMean.train <- mean((train[,9] - pred)^2)
  pred <- predict(BagMod,test[,1:8],type="response")
  BagMean.test <- mean((test[,9] - pred)^2)
  Bag.err <- c(BagMean.train,BagMean.test)
    
  # Trees
  tree.boston=tree(TotalValue ~ ., data=train)
  cv.boston = cv.tree(tree.boston)
  prune.boston = prune.tree(tree.boston)
  trees.pred.train = predict(tree.boston)
  trees.err.train = mean((train[,9] - trees.pred.train)^2)
  trees.pred.test <- predict(tree.boston,newdata = test[,1:8])
  trees.err.test <- mean((test[,9] - trees.pred.test)^2)
  trees.err <- c(trees.err.train,trees.err.test)
  
  errmat[i,] = c(Bag.err, trees.err)
}

# plots
labels = c(rep("BagTraining",n), rep("BagTesting",n),
           rep("TreesTraining",n), rep("TreesTesting",n))
err = sqrt(c(errmat))
boxplot(err~labels,ylab="Misclassification Rate")

