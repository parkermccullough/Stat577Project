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
DF = read.csv("DF.csv")
summary(DF)
head(DF)

# Start Analysis

