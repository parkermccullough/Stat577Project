# STAT 577 Final Project Experimental Script
# 2017-04-20

############### SQL DF creation
# Preliminaries

# Load libraries
library(sqldf)
library(MASS)
library(class)
library(MASS)
library(tree)
require(randomForest)
library(gbm)

# First step is to create the necessary DF we need for analysis
# Data was preprocessed using python with Pandas library, see code/.py
# Load the data from CSV files, Change to absolute path of data
setwd("~/Google Drive/UT Work/STAT 577/Assignments/Project/source/data")

# load inital data files
cust = read.csv(file="transaction_data.csv", header=TRUE)
demo = read.csv(file="demo_ToCategorical.csv", header=TRUE)


stmt <- "SELECT household_key,DAY,STORE_ID,RETAIL_DISC,TRANS_TIME,
WEEK_NO,SALES_VALUE
FROM cust
GROUP BY household_key"
trans1 <- sqldf(stmt)

DF = merge(demo, trans1,by="household_key")
DF$TRANS_TIME_Cat <- cut(DF$TRANS_TIME,
                         breaks=c(56, 1200, 1201, 2354),
                         labels=c("morning","afternoon","evening"))
DF$TRANS_TIME=NULL
DF = na.omit(DF)
names = c('AGE_DESC','MARITAL_STATUS_CODE','INCOME_DESC',
          'HOMEOWNER_DESC','HH_COMP_DESC','HOUSEHOLD_SIZE_DESC',
          'KID_CATEGORY_DESC','DAY','STORE_ID','WEEK_NO',
          'TRANS_TIME_Cat')
DF[,names] = lapply(DF[,names],factor)

write.csv(DF,file="DF.csv")

############### End SQL DF creation


############### Initial Data Analysis
# Box plots, variable by variable trends, etc
DF = read.csv("DF.csv")


############### End Initial Data Analysis

