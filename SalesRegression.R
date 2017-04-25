
# Predicting Sales Using Linear Regression
## A Case Study
### By
## Roshanak Akram
## Nooshin Hamidian
## Parker McCullough
## Varisara Tansakul
### 20 April 2017

# Preliminaries

# Load Libraries
library(sqldf)
library(corrplot)
library(MASS)
library(glmnet)
library(class)
library(leaps)
library(MASS)
library(tree)
require(randomForest)
library(gbm)
library(DMwR)

# Data was preprocessed using python with Pandas library, see code/*.py

### IMPORTANT: MAKE SURE TO BE IN data/ directory
# Load the data from CSV files, Change to absolute path of data
cust = read.csv(file="data/transaction_data.csv", header=TRUE)

demo = read.csv(file="data/demo_ToCategorical.csv", header=TRUE)

stmt <- "SELECT household_key,DAY,STORE_ID,RETAIL_DISC,TRANS_TIME,WEEK_NO,COUPON_DISC,
COUPON_MATCH_DISC,
SUM(SALES_VALUE) AS TotalValue
FROM cust
GROUP BY household_key"
trans1 <- sqldf(stmt)

DF = merge(demo, trans1,by="household_key")
DF$TRANS_TIME_Cat <- cut(DF$TRANS_TIME,
                         breaks=c(56, 1200, 1201, 2354),
                         labels=c("1","2","3"))
write.csv(DF,file="data/DF.csv")

DF = read.csv("data/DF.csv")

DF = na.omit(DF)

DF$household_key = as.numeric(DF$household_key)
DF$AGE_DESC = as.numeric(DF$AGE_DESC)
DF$MARITAL_STATUS_CODE = as.numeric(DF$MARITAL_STATUS_CODE)
DF$INCOME_DESC = as.numeric(DF$INCOME_DESC)
DF$HOMEOWNER_DESC = as.numeric(DF$HOMEOWNER_DESC)
DF$HH_COMP_DESC = as.numeric(DF$HH_COMP_DESC)
DF$HOUSEHOLD_SIZE_DESC = as.numeric(DF$HOUSEHOLD_SIZE_DESC)
DF$KID_CATEGORY_DESC = as.numeric(DF$KID_CATEGORY_DESC)
DF$DAY = as.numeric(DF$DAY)
DF$STORE_ID = as.numeric(DF$STORE_ID)
DF$RETAIL_DISC = as.numeric(DF$RETAIL_DISC)
DF$TRANS_TIME = as.numeric(DF$TRANS_TIME)
DF$WEEK_NO = as.numeric(DF$WEEK_NO)
#DF$TRANS_TIME_Cat =  as.numeric(DF$TRANS_TIME_Cat)
str(DF) 
M = cor(DF)
corrplot(M, method = "circle")

DF = DF[, c(2:9, 17)]
Y_original = DF$TotalValue

outlier.scores <-lofactor(DF, k=200)
outlier <- order(outlier.scores, decreasing=T)[1:50]
DF <- DF[-c(outlier),]

write.csv(DF,file="data/DF_wo_outliers.csv")


# Start here if DF.csv is already created
DF = read.csv("data/DF_wo_outliers.csv")
Y_outlier = DF$TotalValue

plot(Y_original,type="l",col="black")
lines(Y_outlier,col="red")


YY = as.numeric(DF[,9]); YY = YY - mean(YY) # center the response
XX = as.matrix(DF[,-9]); XX = scale(XX,center=T,scale=T) # center the predictors
n = nrow(XX)
rep = 20
mse.train.lm = mse.train.bs = mse.train.f = mse.train.b = mse.train.l = mse.train.r = mse.train.en = mse.train.pcr = mse.train.pls = rep(0,rep)
mse.test.lm = mse.test.bs = mse.test.f = mse.test.b = mse.test.l = mse.test.r = mse.test.en =
  mse.test.pcr = mse.test.pls = rep(0,rep)

# loop
set.seed(1)
for(i in 1:rep)
{
  train_index = sample(1:n,n/2,replace = FALSE)  # randomly select the indices of the training set
  X = XX[train_index,]  # trainining X
  Y = YY[train_index]  # training Y
  X_test = XX[-c(train_index),] # Test X
  Y_test = YY[-c(train_index)] # Test Y
  train_data = data.frame(X,Y)
  test_data = data.frame(X_test,Y_test)
  
  # LM
  lm.fit = lm(Y~., data = train_data) #
  pred.train.lm = predict(lm.fit)
  pred.test.lm = predict(lm.fit,newdata = test_data[,-10])
  mse.train.lm[i] = sqrt(sum((Y - pred.train.lm)^2)/length(Y))
  mse.test.lm[i] = sqrt(sum((Y_test - pred.test.lm)^2)/length(Y_test))
  
  #Best Subset
  fitbsub = regsubsets(x=X,y=Y,nbest = 1, nvmax = 10)
  rs = summary(fitbsub)
  plot(rs$bic, xlab="Parameter", ylab="BIC",type = 'l')
  bs.fit = lm(Y ~ AGE_DESC+MARITAL_STATUS_CODE+INCOME_DESC,data=train_data )#!!!!!!!
  pred.train.bs = predict(bs.fit)
  pred.test.bs = predict(bs.fit,newdata = test_data[,c(3,4,5)])#!!!!!!!
  mse.train.bs[i] = sqrt(sum((Y - pred.train.bs)^2)/length(Y))
  mse.test.bs[i] = sqrt(sum((Y_test - pred.test.bs)^2)/length(Y_test))
  
  # forward step-wise
  #fitf0 = lm(Y~1,data=train_data) # fit the model with only the intercept
  #fitf = stepAIC(fitf0,lpsa~.,direction="forward",data=prostate,k=log(nrow(prostate)))
  #pred.train.f = predict(fitf)
  #pred.train.f = predict(fitf,newdata = test_data[,-9])
  #mse.train.f[i] = sum(Y - pred.train.f)^2/length(Y)
  #mse.test.f[i] = sum(Y_test - pred.test.f)^2/length(Y_test)
  
  #backward step-wise
  fitb0 = lm(Y~.,data=train_data)
  fitb = stepAIC(fitb0,direction="backward",data=DF,k=log(nrow(DF)),trace=FALSE)
  pred.train.b  = predict(fitb)
  pred.test.b = predict(fitb,newdata = test_data[,-10])
  mse.train.b[i] = sqrt(sum((Y - pred.train.b)^2)/length(Y))
  mse.test.b[i] = sqrt(sum((Y_test - pred.test.b)^2)/length(Y_test))
  
  #lasso
  cvfitl = cv.glmnet(x=X,y=Y,family="gaussian",alpha=1,standardize=FALSE)
  pred.train.l = predict(cvfitl, newx = X, s = "lambda.min") 
  pred.test.l = predict(cvfitl, newx = X_test, s = "lambda.min") 
  mse.train.l[i] = sqrt(sum((Y - pred.train.l)^2)/length(Y))
  mse.test.l[i] = sqrt(sum((Y_test - pred.test.l)^2)/length(Y_test))
  
  # Ridge
  cvfitr = cv.glmnet(x=X,y=Y,family="gaussian",alpha=0,standardize=FALSE)
  pred.train.r = predict(cvfitr, newx = X, s = "lambda.min") 
  pred.test.r = predict(cvfitr, newx = X_test, s = "lambda.min") 
  mse.train.r[i] = sqrt(sum((Y - pred.train.r)^2)/length(Y))
  mse.test.r[i] = sqrt(sum((Y_test - pred.test.r)^2)/length(Y_test))
  
  # E - net
  cvfiten = cv.glmnet(x=X,y=Y,family="gaussian",alpha=0.5,standardize=FALSE)
  pred.train.en = predict(cvfiten, newx = X, s = "lambda.min") 
  pred.test.en = predict(cvfiten, newx = X_test, s = "lambda.min") 
  mse.train.en[i] = sqrt(sum((Y - pred.train.en)^2)/length(Y))
  mse.test.en[i] = sqrt(sum((Y_test - pred.test.en)^2)/length(Y_test))
  
  # PC regression 
  pcr.fit=pcr(Y~X,scale=TRUE,ncomp = 5)
  pred.train.pcr = predict(pcr.fit, ncomp = 5, newdata = X)
  pred.test.pcr = predict(pcr.fit, ncomp = 5, newdata = X_test)
  mse.train.pcr[i] = sqrt(sum((Y - pred.train.pcr)^2)/length(Y))
  mse.test.pcr[i] = sqrt(sum((Y_test - pred.test.pcr)^2)/length(Y_test))
  
  # PLS regression
  pls.fit=plsr(Y~X,scale=TRUE,ncomp = 5)
  pred.train.pls = predict(pls.fit, ncomp = 5, newdata = X)
  pred.test.pls = predict(pls.fit, ncomp = 5, newdata = X_test)
  mse.train.pls[i] = sqrt(sum((Y - pred.train.pls)^2)/length(Y))
  mse.test.pls[i] = sqrt(sum((Y_test - pred.test.pls)^2)/length(Y_test))
}

mse.train = c(mse.train.lm,mse.train.bs, mse.train.b, mse.train.l,mse.train.r,mse.train.en,
              mse.train.pcr,mse.train.pls)
mse.test = c(mse.test.lm, mse.test.bs, mse.test.b, mse.test.l, mse.test.r, mse.test.en, 
             mse.test.pcr, mse.test.pls)
indicator = c(rep("LM",rep),rep("BS",rep),rep("BackWard",rep),rep("Lasso",rep),rep("Ridge",rep),
              rep("ENET",rep),rep("PCR",rep),rep("PLSR",rep))
par(mfrow=c(1,2))
colours <- c("red", "orange", "blue", "yellow", "green", "hotpink", "brown","purple")
boxplot(mse.train~as.factor(indicator),main="Training RMSE",ylab="RMSE", las=2, col=colours)
boxplot(mse.test~as.factor(indicator),main="Test RMSE",ylab="RMSE", las=2, col=colours)

#-------------------------------------------------------------------------
# Boosting
#-------------------------------------------------------------------------
errmat = matrix(0,20,2)
for(i in 1:20)
{
  DF.tr = data.frame(Y,DF[train_index,-10])
  bs.DF=gbm(Y~.,data=DF.tr,distribution="gaussian",n.trees=5000,interaction.depth=4)
  summary(bs.DF)
  
  yhat.boost=predict(bs.DF,newdata=test_data,n.trees=5000)
  sqrt(mean((yhat.boost-Y_test)^2)) # MSE
  #bs.DF=gbm(Y~.,data=DF.tr,distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F)
  #yhat.boost=predict(bs.DF,newdata=test_data,n.trees=5000)
  bs.tserr = sqrt(mean((yhat.boost-Y_test)^2)) # MSE
  bs.trerr = sqrt(mean((yhat.boost-Y)^2))
  bs.err = c(bs.trerr, bs.tserr)
  
  errmat[i,] = c(bs.err) 
}

labels = as.factor(c(rep("BSTr",20),rep("BSTs",20)))
err = c(errmat)
par(mfrow=c(1,1))
boxplot(err~labels,ylab="misclassification rate", las=2)

######################
# Bagging and Trees
######################

# Load Data
DF = read.csv("data/DF_wo_outliers.csv")
DF$X=NULL
n = 20
errmat = matrix(0,n,4)
set.seed(2)

for (i in 1:n){
  # Split data into Training and Test sample
  trainIndex <- sample(1:nrow(DF),375)
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
