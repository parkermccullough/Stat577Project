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
DF = read.csv("data/DF.csv")
DF$household_key=NULL
DF$X=NULL
summary(DF)
head(DF)
dim(DF)
head(DF)

# Start Analysis 
# See homework 4 for best place to start.

errmat = matrix(0,20,2)
set.seed(2)

for (i in 1:20){
    # Split data into Training and Test sample
  trainIndex <- sample(1:nrow(DF),400)
  test <- DF[-trainIndex,]
  train <- DF[trainIndex,]
    
  # Bagging
  BagMod <- randomForest(TotalValue~.,data=train,mtry=14)
  pred <- predict(BagMod,type = "response")
  BagMean.train <- mean((train[,14] - pred)^2)
  pred <- predict(BagMod,test,type="response")
  BagMean.test <- mean((train[,14] - pred)^2)
  Bag.err <- c(BagMean.train,BagMean.test)
    
  errmat[i,] = c(BagMean.train,BagMean.test)
}

# plots
labels = c(rep("BagTraining",20),
                     rep("BagTesting",20))
err = c(errmat)
boxplot(err~labels,ylab="Misclassification Rate")

Total=sum(DF$TotalValue)
Total
sqrt(errmat)/Total*100
mean(sqrt(errmat[,1]))
mean(sqrt(errmat[,2]))
