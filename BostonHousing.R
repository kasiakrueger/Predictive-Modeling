#
# Boston Housing Study
# 

library(caret)
library(corrplot) 
library(pls)
library(mlbench)
library(e1071)

# load data set from mlbench package
data(BostonHousing)
str(BostonHousing)

# crim per capita crime rate by town
# zn proportion of residential land zoned for lots over 25,000 sq.ft
# indus proportion of non-retail business acres per town
# chas Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
# nox nitric oxides concentration (parts per 10 million)
# rm average number of rooms per dwelling
# age proportion of owner-occupied units built prior to 1940
# dis weighted distances to five Boston employment centers
# rad index of accessibility to radial highways
# tax full-value property-tax rate per USD 10,000
# ptratio pupil-teacher ratio by town
# b 1000(B − 0.63)^2 where B is the proportion of blacks by town
# lstat percentage of lower status of the population
# medv median value of owner-occupied homes in USD 1000’s

# medv is the outcome variable.

# scatter plots of raw predictors vs outcome (medv)
chasIndex <- grep("chas", colnames(BostonHousing))
medvIndex <- grep("medv", colnames(BostonHousing))
Pnames <- colnames(BostonHousing[,-c(medvIndex, chasIndex)])
par(mfrow = c(2,2))
for (predictor in Pnames) {
  predictors <- as.vector(unlist(get("BostonHousing")[predictor]))
  plot(BostonHousing$medv, predictors, 
          main = paste("Plot of", predictor, "vs. medv"),
          ylab = predictor,
          xlab = "medv"
  )
}

### data pre-processing ###

# check for NA's in outcome
medvNAs <- is.na(BostonHousing$cmedv)
medvNAs

# check for NA's in predictors
NAs <- is.na(BostonHousing)
NAsTrue <- grep("TRUE", NAs)
NAsTrue

# no missing data, no need for imputation

# histograms of raw predictors
chasIndex <- grep("chas", colnames(BostonHousing))
medvIndex <- grep("medv", colnames(BostonHousing))
Pnames <- colnames(BostonHousing[, -c(medvIndex, chasIndex)])
par(mfrow = c(2, 2))
for (predictor in Pnames) {
  predictors <- as.vector(unlist(get("BostonHousing")[predictor]))
  hist(predictors,
       main = paste(predictor),
       xlim =c(min(predictors), max(predictors)),
       xlab = predictor,
       ylab = "Frequency"
  )
}

# check skewness of raw predictors
skew <- apply(BostonHousing[, -chasIndex], 2, skewness)
skew

# log transform crim, nox, dis, rad, tax, lstat, and medv
transVars <- c("crim", "nox", "dis", "rad", "tax", "lstat", "medv")
for (var in transVars) {
  varIndex <- grep(var, colnames(BostonHousing))
  BostonHousing[,varIndex] <- log(BostonHousing[,varIndex])
}

# skewness of log transformed predictors
skew <- apply(BostonHousing[, -chasIndex], 2, skewness)
skew

# histograms of skew transformed predictors
chasIndex <- grep("chas", colnames(BostonHousing))
medvIndex <- grep("medv", colnames(BostonHousing))
Pnames <- colnames(BostonHousing[, -c(medvIndex, chasIndex)])
par(mfrow = c(2, 2))
for (predictor in Pnames) {
  predictors <- as.vector(unlist(get("BostonHousing")[predictor]))
  hist(predictors,
       main = paste(predictor),
       xlim =c(min(predictors), max(predictors)),
       xlab = predictor,
       ylab = "Frequency"
  )
}

# boxplots to check for outliers
chasIndex <- grep("chas", colnames(BostonHousing))
medvIndex <- grep("medv", colnames(BostonHousing))
Pnames <- colnames(BostonHousing[,-c(medvIndex, chasIndex)])
par(mfrow = c(2,2))
for (predictor in Pnames) {
  predictors <- as.vector(unlist(get("BostonHousing")[predictor]))
  boxplot(predictors, 
          main = paste("BoxPlot of", predictor),
          ylab = "Value"
  )
}

# examination of the data finds all outliers are valid data

# correlation plot
medvIndex <- grep("medv", colnames(BostonHousing))
chasIndex <- grep("chas", colnames(BostonHousing))
corrplot::corrplot(cor(BostonHousing[-c(medvIndex, chasIndex)]), order="hclust")

# check for correlated predictors
cor90 <- findCorrelation(cor(BostonHousing[,-c(medvIndex, chasIndex)]), cutoff=0.90)
cor90
colnames(BostonHousing[,-c(medvIndex, chasIndex)])[cor90]

# correlation between predictors and outcome
corrValues <- apply(BostonHousing[,-c(medvIndex, chasIndex)],
      MARGIN  = 2,
      FUN = function(x ,y) cor(x, y),
      y = BostonHousing$medv)
corrValues

# plot predictors against each other
par(mfrow = c(1,1))
pairs(BostonHousing[,-c(medvIndex, chasIndex)])

# check for near zero variance predictors
nearZeroVar(BostonHousing[,-c(medvIndex, chasIndex)])

# no near zero predictors

### linear model ###
set.seed(0)
lmMod <- lm(medv ~ ., data = BostonHousing)
summary(lmMod)

# coefficients for crim, zn, nox, dis, tax, ptratio, and lstat are negative

# the following parameters decrease medv
# crim per capita crime 
# zn proportion of residential land zoned for lots over 25,000 sq.ft  value
# dis weighted distances to five Boston employment centers 
# nox nitric oxides concentration (parts per 10 million)
# tax full-value property-tax rate per USD 10,000 
# ptratio pupil-teacher ratio by town 
# lstat percentage of lower status of the population 

# the following parameters increase medv
# indus proportion of non-retail business acres per town
# chas Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
# rm average number of rooms per dwelling
# age proportion of owner-occupied units built prior to 1940
# rad index of accessibility to radial highways
# b 1000(B − 0.63)^2 where B is the proportion of blacks by town

# transform the predictors with BoxCox no PCA 
trans <- preProcess(BostonHousing[,-medvIndex],
    method = c("BoxCox", "center", "scale"))
trans

# apply the non PCA transformation
BHtrans <- predict(trans, BostonHousing)
head(BHtrans)


### explore the data ###

# split outcome and predictors
medvIndex <- grep("medv", colnames(BHtrans))
BHOutcome <- BHtrans$medv
BHPredictors <- BHtrans[,-medvIndex] 

# histograms of BoxCox transformed variables
chasIndex <- grep("chas", colnames(BHPredictors))
Pnames <- colnames(BHPredictors[,-chasIndex])
par(mfrow = c(2, 2))
for (predictor in Pnames) {
  predictors <- as.vector(unlist(get("BHPredictors")[predictor]))
  hist(predictors,
       main = paste(predictor),
       xlim =c(min(predictors), max(predictors)),
       xlab = predictor,
       ylab = "Frequency"
  )
}

# boxplots of factor predictor against the outcome variable
par(mfrow = c(1,1))
boxplot(BHOutcome ~ BHtrans$chas, data = BHtrans,
    ylab = "medv",
    xlab = "chas")

### spend the data ###

# select 75% for training and 25% for testing
set.seed(0)
trainIndex <- sample(1:length(BHOutcome), length(BHOutcome)*0.75, replace = FALSE)
head(trainIndex)
length(trainIndex)
BHPredictorsTrain <- BHPredictors[trainIndex,]
BHPredictorsTest <- BHPredictors[-c(trainIndex),]
BHOutcomeTrain <- BHOutcome[trainIndex]
BHOutcomeTest <- BHOutcome[-c(trainIndex)]
medv <- BHOutcomeTrain
BHTrainSet <- cbind(medv, BHPredictorsTrain)
medv <- BHOutcomeTest
BHTestSet <- cbind(medv, BHPredictorsTest)

# correlation between all components
corr <- cor(BHPredictors[,-chasIndex])
corr

# correlation plot
chasIndex <- grep("chas", colnames(BHPredictors))
corrplot::corrplot(cor(BHPredictors[,-chasIndex]), order="hclust")


### model building ###

# GLM with PCA data
medvIndex <- grep("medv", colnames(BostonHousing))
trans <- preProcess(BostonHousing[,-medvIndex],
                    method = c("BoxCox", "center", "scale", "pca"))
trans

# apply the transformation
BHtransPCA <- predict(trans, BostonHousing)
head(BHtransPCA)
# eight components provide 95% of the variance

# calculate total variance explained by each component
medvIndex <- grep("medv", colnames(BHtransPCA))
chasIndex <- grep("chas", colnames(BHtransPCA))
var <- sapply(BHtransPCA[,-c(medvIndex, chasIndex)], "var")
var <- var / sum(var)
var

# scree plot of PCA data
qplot(c(1:8), var) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("Variance Explained") +
  ggtitle("Scree Plot") +
  ylim(0, 1)

medvIndex <- grep("medv", colnames(BHtransPCA))
BHOutcomePCA <- BHtransPCA$medv
BHPredictorsPCA <- BHtransPCA[,-medvIndex] 
BHPredictorsPCATrain <- BHPredictorsPCA[trainIndex,]
BHPredictorsPCATest <- BHPredictorsPCA[-c(trainIndex),]
BHOutcomePCATrain <- BHOutcomePCA[trainIndex]
BHOutcomePCATest <- BHOutcomePCA[-c(trainIndex)]
medv <- BHOutcomePCATrain
BHTrainPCASet <- cbind(medv, BHPredictorsPCATrain)
medv <- BHOutcomePCATest
BHTestPCASet <- cbind(medv, BHPredictorsPCATest)

# histograms of pca components
chasIndex <- grep("chas", colnames(BHPredictorsPCA))
Pnames <- colnames(BHPredictorsPCA[,-chasIndex])
par(mfrow = c(2, 2))
for (predictor in Pnames) {
  predictors <- as.vector(unlist(get("BHPredictorsPCA")[predictor]))
  hist(predictors,
       main = paste(predictor),
       xlim =c(min(predictors), max(predictors)),
       xlab = predictor,
       ylab = "Frequency"
  )
}

set.seed(0)
glmMod <- train(medv ~ ., data = BHTrainPCASet,
                method = "glm", 
                trControl = trainControl(method = "repeatedcv", 
                number = 3, repeats = 5))
glmMod
glmMod$finalModel

plot(varImp(glmMod), 10, main="GLM-Importance")

# predict on test set
set.seed(0)
glmPred <- predict(glmMod, newdata = BHTestPCASet)
glmValues <- postResample(pred = glmPred, obs = BHOutcomePCATest) 
glmValues

p <- ggplot(BHTestSet, aes(x = medv, y = glmPred))
p +
  labs(title = "GLM Model", x = "Actual", y = "Predicted") +
  geom_point() +
  stat_smooth(method = glm, formula = 'y~x')


# knn

# find optimum K value
set.seed(0)
knnMod <- train(medv ~ ., data = BHTrainSet, 
                 method = "knn",
                 # Center and scaling will occur for new predictions too
                 #preProc = c("center", "scale"),
                 tuneGrid = data.frame(.k = 1:20),
                 trControl = trainControl(method = "repeatedcv", repeats = 5))
knnMod
plot(knnMod)
knnMod$finalModel

# k = 3 is optimum value

set.seed(0)
knnMod <- train(medv ~ ., data = BHTrainSet, 
                method = "knn",
                tuneGrid = data.frame(.k = 2:5),
                trControl = trainControl(method = "repeatedcv", 
                number = 3, repeats = 5))
knnMod

plot(varImp(knnMod), 10, main="KNN-Importance")

# predict on testing set
set.seed(0)
knnPred <- predict(knnMod, newdata = BHTestSet)
knnValues <- postResample(pred = knnPred, obs = BHOutcomeTest) 
knnValues

p <- ggplot(BHTestSet, aes(x = medv, y = knnPred))
p +
    labs(title = "KNN Model", x = "Actual", y = "Predicted") +
    geom_point() +
    stat_smooth(method = glm, formula = 'y~x')


# SVM 
set.seed(0)
svmMod <- train(medv ~ ., data = BHTrainSet, 
    method = "svmRadial",
    tuneLength = 10,
    trControl = trainControl(method = "repeatedcv", 
    number = 3, repeats = 5))
svmMod
plot(svmMod)

plot(varImp(svmMod), 10, main="SVM-Importance")
svmMod$finalModel

# line plot of the average performance
plot(svmMod, scales = list(x = list(log = 2)))

# predict on testing set
set.seed(0)
svmPred <- predict(svmMod, newdata = BHTestSet)
svmValues <- postResample(pred = svmPred, obs = BHOutcomeTest) 
svmValues

p <- ggplot(BHTestSet, aes(x = medv, y = svmPred))
p +
    labs(title = "SVM Model", x = "Actual", y = "Predicted") +
    geom_point() +
    stat_smooth(method = glm, formula = 'y~x')


# PLS
set.seed(0)
plsMod <- train(medv ~ ., data = BHTrainSet,
    method = "pls", 
    trControl = trainControl(method = "repeatedcv", 
    number = 3, repeats = 5))
plsMod
plot(plsMod)
plsMod$finalModel

plot(varImp(plsMod), 10, main="PLS-Importance")

# predict on testing set
set.seed(0)
plsPred <- predict(plsMod, BHTestSet, ncomp = 8)
plsValues <- postResample(pred = plsPred, obs = BHOutcomeTest) 
plsValues

p <- ggplot(BHTestSet, aes(x = medv, y = plsPred))
p +
  labs(title = "PLS Model", x = "Actual", y = "Predicted") +
  geom_point() +
  stat_smooth(method = glm, formula = 'y~x')


# elastic net 
enetGrid = expand.grid(.lambda=seq(0,1,length=20), .fraction=seq(0.05, 1.0, length=20))
set.seed(0)
enetMod = train(medv ~ ., data = BHTrainSet,
                method="enet",
                tuneGrid = enetGrid,
                trControl=trainControl(method="repeatedcv",
                number = 3, repeats=5))
enetMod
plot(enetMod)
enetMod$finalModel

plot(varImp(enetMod), 10, main="ENET-Importance")

# predict on testing set
set.seed(0)
enetPred <- predict(enetMod, BHTestSet)
enetValues <- postResample(pred = enetPred, obs = BHOutcomeTest) 
enetValues

p <- ggplot(BHTestSet, aes(x = medv, y = enetPred))
p +
  labs(title = "E-net Model", x = "Actual", y = "Predicted") +
  geom_point() +
  stat_smooth(method = glm, formula = 'y~x')


# MARS
marsGrid = expand.grid(.degree = 1:2, .nprune = 2:20)
set.seed(0)
marsMod = train(x = BHPredictorsTrain, y = BHOutcomeTrain, 
                method = "earth", 
                tuneGrid = marsGrid)
marsMod
plot(marsMod)
marsMod$finalModel

plot(varImp(marsMod), 10, main="MARS-Importance")

# predict on testing set
set.seed(0)
marsPred <- predict(marsMod, BHTestSet)
marsValues <- postResample(pred = marsPred, obs = BHOutcomeTest) 
marsValues

p <- ggplot(BHTestSet, aes(x = medv, y = marsPred))
p +
  labs(title = "MARS Model", x = "Actual", y = "Predicted") +
  geom_point() +
  stat_smooth(method = glm, formula = 'y~x')

marsMod$finalModel

# MARS importance plot
plot(varImp(marsMod), 10, main="MARS-Importance") 

# compare models based on their cross-validation statistics.

# create a resamples object from the models:
resamp <- resamples(list(KNN = knnMod, 
                         SVM = svmMod, 
                         GLM = glmMod, 
                         PLS = plsMod,
                         ENET = enetMod))
summary(resamp)

# compare testing results
table <- rbind(knnValues, 
               svmValues, 
               glmValues, 
               plsValues, 
               enetValues,
               marsValues)
table

# SVM has the smallest RMSE and largest R-squared


#
# bin outcome variable, medv, and use categorical models
#

# histogram of medv
hist(BHOutcome,
     main = "Histogram of medv",
     xlab = "medv",
     ylab = "Frequency"
)

# check skewness
skew <- skewness(BHOutcome)
skew

if (0) {
# bin medv into 3 equal groups, low, medium, and high price
medvCat <- rep(1, length(BHOutcome))
medv <- BHtrans$medv
BHOutcome <- as.data.frame(cbind(medv, medvCat))
medvSort <- sort(BHOutcome$medv)
split1 <- medvSort[169]
split2 <- medvSort[337]
for (i in 1:nrow(BHOutcome)) {
  if (BHOutcome[i, "medv"] <= split1) {
    BHOutcome[i,"medvCat"] <- 0
  }
  if (BHOutcome[i, "medv"] >= split2) {
    BHOutcome[i,"medvCat"] <- 2
  }
}
BHOutcome$medvCat <- factor(BHOutcome$medvCat, labels = c("low", "medium", "high"))
numLow <- nrow(BHOutcome[which (BHOutcome$medvCat == "low"),])
numLow
numMed <- nrow(BHOutcome[which (BHOutcome$medvCat == "medium"),])
numMed
numHigh <- nrow(BHOutcome[which (BHOutcome$medvCat == "high"),])
numHigh
}

# bin medv into 2 equal groups, low and high price
medvCat <- rep(0, length(BHOutcome))
medv <- BHtrans$medv
BHOutcome <- as.data.frame(cbind(medv, medvCat))
medvSort <- sort(BHOutcome$medv)
split <- medvSort[252]
for (i in 1:nrow(BHOutcome)) {
  if (BHOutcome[i, "medv"] > split) {
    BHOutcome[i,"medvCat"] <- 1
  }
}
BHOutcome$medvCat <- factor(BHOutcome$medvCat, labels = c("low", "high"))
numLow <- nrow(BHOutcome[which (BHOutcome$medvCat == "low"),])
numLow
numHigh <- nrow(BHOutcome[which (BHOutcome$medvCat == "high"),])
numHigh
head(BHOutcome)
chasIndex <- grep("chas", colnames(BHPredictors))


# training control
ctrl <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

# split outcome into training and testing as previous
BHOutcomeTrain <- BHOutcome[trainIndex,]
BHOutcomeTest <- BHOutcome[-c(trainIndex),]

# training control
ctrl <- trainControl(summaryFunction = twoClassSummary,
                     #method = "LGOCV",
                     classProbs = TRUE,
                     savePredictions = TRUE)


# Logistic Regression Model

levels(BHOutcome$medvCat)

set.seed(1000)
lrFit = train(BHPredictorsTrain[-chasIndex], BHOutcomeTrain$medvCat, 
              method = "glm", 
              metric = "ROC", 
              trControl = ctrl)
lrFit

# predict on test set
lrPred = predict(lrFit, BHPredictorsTest[-chasIndex])
lrValues <- postResample(pred = lrPred, obs =  BHOutcomeTest$medvCat) 
lrValues

# test set
confusionMatrix(data = lrPred,
                reference = BHOutcomeTest$medvCat)

# training set
confusionMatrix(data = lrFit$pred$pred,
                reference = lrFit$pred$obs)

lrImpSim <- varImp(lrFit, scale = FALSE)
lrImpSim
par(mfrow = c(1,1))
plot(lrImpSim, top = 5, scales = list(y = list(cex = .95)))

# AUC
library(pROC)
lrRoc <- roc(response = lrFit$pred$obs,
             predictor = lrFit$pred$high,
             levels = rev(levels(lrFit$pred$obs)))
plot(lrRoc, legacy.axes = TRUE, main = "Logistic Regression")
lrAUC <- auc(lrRoc)
lrAUC

# LDA
library(MASS)
set.seed(1000)
ldaModel <- lda(BHPredictorsTrain[-chasIndex], 
                grouping = BHOutcomeTrain$medvCat)
ldaModel
summary(ldaModel)

set.seed(1000)
ldaFit <- train(x = as.data.frame(BHPredictorsTrain[-chasIndex]), 
                y = BHOutcomeTrain$medvCat, 
                method = "lda",
                preProc = c("center", "scale"),
                metric = "ROC",
                trControl = ctrl)
ldaFit

# predict on test set
ldaPred = predict(ldaFit, BHPredictorsTest[-chasIndex])
ldaValues <- postResample(pred = ldaPred, obs = BHOutcomeTest$medvCat) 
ldaValues

# test set
confusionMatrix(data = ldaPred,
                reference = BHOutcomeTest$medvCat)

# training set
confusionMatrix(data = ldaFit$pred$pred,
                reference = ldaFit$pred$obs)

#ldaImpSim <- varImp(ldaFit, scale = FALSE)
#ldaImpSim
#plot(ldaImpSim, top = 5, scales = list(y = list(cex = .95)))

# AUC
ldaRoc <- roc(response = ldaFit$pred$obs,
              predictor = ldaFit$pred$high,
              levels = rev(levels(ldaFit$pred$obs)))
plot(ldaRoc, legacy.axes = TRUE, main = "Linear Discriminant Analysis")
ldaAUC <- auc(ldaRoc)
ldaAUC

# PLSDA 

set.seed(1000)
plsdaModel <- plsda(BHPredictorsTrain[-chasIndex], BHOutcomeTrain$medvCat,
                    scale = TRUE,
                    ncomp = 10)
plsdaModel

set.seed(1000)
plsFit <- train(BHPredictorsTrain[-chasIndex], BHOutcomeTrain$medvCat,
                method = "pls",
                tuneGrid = expand.grid(.ncomp = 1:10),
                preProc = c("center","scale"),
                metric = "ROC",
                trControl = ctrl)
plsFit
plot(plsFit)

plsImpSim <- varImp(plsFit, scale = FALSE)
plsImpSim
plot(plsImpSim, top = 5, scales = list(y = list(cex = .95)))

# predict on testing
plsPred <- predict(plsFit, BHPredictorsTrain[-chasIndex])
plsValues <- postResample(pred = plsPred, obs = BHOutcomeTrain$medvCat)
plsValues

# test set
confusionMatrix(data = plsPred,
                reference = BHOutcomeTest$medvCat)

# training set
confusionMatrix(data = plsFit$pred$pred,
                reference = plsFit$pred$obs)

# AUC
plsRoc <- roc(response = plsFit$pred$obs,
              predictor = plsFit$pred$high,
              levels = rev(levels(plsFit$pred$obs)))
plot(plsRoc, legacy.axes = TRUE, main = "Partial Least Squares")
plsAUC <- auc(plsRoc)


# Nearest shrunken Centroids
nscGrid = expand.grid(.threshold = 1:10)

set.seed(1000)
nscFit = train(BHPredictorsTrain[-chasIndex], BHOutcomeTrain$medvCat, 
               method = "pam", 
               preProc = c("center","scale"),
               tuneGrid = nscGrid, 
               metric = "ROC", 
               trControl = ctrl)
nscFit
plot(nscFit)

nscImpSim <- varImp(nscFit, scale = FALSE)
nscImpSim
plot(nscImpSim, top = 5, scales = list(y = list(cex = .95)))

# predict on testing
nscPred <- predict(nscFit, BHPredictorsTrain[-chasIndex])
nscValues <- postResample(pred = nscPred, obs = BHOutcomeTrain$medvCat)
nscValues

# test set
confusionMatrix(data = nscPred,
                reference = BHOutcomeTest$medvCat)

# training set
confusionMatrix(data = nscFit$pred$pred,
                reference = nscFit$pred$obs)

# AUC
nscRoc <- roc(response = nscFit$pred$obs,
              predictor = nscFit$pred$high,
              levels = rev(levels(nscFit$pred$obs)))
plot(nscRoc, legacy.axes = TRUE, main = "Nearest Shrunken Centroids")
nscAUC <- auc(nscRoc)

############ GLM NET MODEL ############ 

library(glmnet)


#glmnetModel <- glmnet(x = bioTrain,
# y = injuryTrain,
# family = "binomial")


ctrl <- caret::trainControl(method = "LGOCV",
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            ##index = list(simulatedTest[,1:4]),
                            savePredictions = TRUE)

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(0, .5, length = 5))
set.seed(1000)
glmnMod <- train(x = BHPredictorsTrain[-chasIndex], y = BHOutcomeTrain$medvCat,
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 preProc = c("center", "scale"),
                 #metric = "ROC",
                 trControl = ctrl)
glmnMod
plot(glmnMod)

# predict on testing
glmnPred <- predict(glmnMod, BHPredictorsTrain[-chasIndex])

#test set values
glmnValues <- postResample(pred = glmnPred, obs = BHOutcomeTrain$medvCat)
glmnValues

# test set
confusionMatrix(data = glmnPred,
                reference = BHOutcomeTrain$medvCat)


# train set values
confusionMatrix(data = glmnMod$pred$pred,
                reference = glmnMod$pred$obs)
glmnImpSim <- varImp(glmnMod, scale = FALSE)
glmnImpSim
plot(glmnImpSim, top = 5, scales = list(y = list(cex = .95)))


# AUC
glmnRoc <- roc(response = glmnMod$pred$obs,
               predictor = glmnMod$pred$high,
               levels = rev(levels(glmnMod$pred$obs)))
plot(glmnRoc, legacy.axes = TRUE, main = "Penalized GLM")

glmnAUC <- auc(glmnRoc)
glmnAUC

# compare testing results
table <- rbind(lrValues,
               ldaValues, 
               plsValues,
               nscValues,
               glmnValues)
table

table <- rbind(lrAUC,
               ldaAUC, 
               plsAUC,
               nscAUC,
               glmnAUC)
table

################################################

# training control
ctrl <- trainControl(summaryFunction = twoClassSummary,
                     method = "LGOCV",
                     classProbs = TRUE,
                     savePredictions = TRUE)

############## MDA ############## 
par(mfrow = c(1, 1))
mdaFit <- train(BHPredictorsTrain[-chasIndex],                  BHOutcomeTrain$medvCat,
                method = "mda",
                metric = "ROC",
                tuneGrid = expand.grid(.subclasses = 1:10),
                trControl = ctrl)
mdaFit
plot(mdaFit)


# predict on testing set
mdaPred <- predict(mdaFit, BHPredictorsTrain[-chasIndex])
mdaValues <- postResample(pred = mdaPred, obs = BHOutcomeTrain$medvCat) 
mdaValues

#test set
confusionMatrix(data = mdaPred,
                reference = BHOutcomeTrain$medvCat)

#train set
confusionMatrix(data = mdaFit$pred$pred,
                reference = mdaFit$pred$obs)

#mdaImpSim <- varImp(mdaFit, scale = FALSE)
#mdaImpSim
#plot(mdaImpSim, top = 5, scales = list(y = list(cex = .95)))

# AUC
mdaPred <- predict(mdaFit, BHPredictorsTrain[-chasIndex], type = "prob")
mdaRoc <- roc(response = mdaFit$pred$obs,
              predictor = mdaFit$pred$high,
              levels = rev(levels(mdaFit$pred$obs)))
plot(mdaRoc, legacy.axes = TRUE)
mdaAUC <- auc(mdaRoc)
mdaAUC

############## NNet ############## 

nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, .3, .5, 1))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (102 + 1) + (maxSize+1)*2) ## 102 is the number of predictors; 2 is the number of classes

#takes 5 minutes:
nnetFit <- train(BHPredictorsTrain[-chasIndex],                  BHOutcomeTrain$medvCat,
                 method = "nnet",
                 metric = "ROC",
                 preProc = c("center", "scale", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit 
plot(nnetFit)

# predict on testing set
nnetPred <- predict(nnetFit, BHPredictorsTrain[-chasIndex])
nnetValues <- postResample(pred = nnetPred, obs = BHOutcomeTrain$medvCat) 
nnetValues
#test set
confusionMatrix(data = nnetPred,
                reference = BHOutcomeTrain$medvCat)

#train set
confusionMatrix(data = nnetFit$pred$pred,
                reference = nnetFit$pred$obs)

# AUC
nnetPred <- predict(nnetFit,  BHPredictorsTrain[-chasIndex], type = "prob")
nnetRoc <- roc(response = nnetFit$pred$obs,
               predictor = nnetFit$pred$high,
               levels = rev(levels(nnetFit$pred$obs)))
plot(nnetRoc, legacy.axes = TRUE)
nnetAUC <- auc(nnetRoc)
nnetAUC

############## FDA ############## 

marsGrid <- expand.grid(.degree = 1:10, .nprune = 2:10)

fdaTuned <- train(BHPredictorsTrain[-chasIndex],                  BHOutcomeTrain$medvCat,
                  method = "fda",
                  # Explicitly declare the candidate models to test
                  tuneGrid = marsGrid,
                  trControl = ctrl)

fdaTuned
plot(fdaTuned)



# predict on testing set
fdaPred <- predict(fdaTuned, BHPredictorsTrain[-chasIndex])
fdaValues <- postResample(pred = fdaPred, obs = BHOutcomeTrain$medvCat) 
fdaValues

#test set
confusionMatrix(data = fdaPred,
                reference = BHOutcomeTrain$medvCat)

#train set
confusionMatrix(data = fdaTuned$pred$pred,
                reference = fdaTuned$pred$obs)



# AUC
fdaPred <- predict(fdaTuned, BHPredictorsTrain[-chasIndex], type = "prob")
fdaRoc <- roc(response = fdaTuned$pred$obs,
              predictor = fdaTuned$pred$high,
              levels = rev(levels(fdaTuned$pred$obs)))
plot(fdaRoc, legacy.axes = TRUE)
fdaAUC <- auc(fdaRoc)
fdaAUC

############## Support Vector Machines ############## 
library(kernlab)
library(caret)
sigmaRangeReduced <- sigest(as.matrix(BHPredictorsTrain[-chasIndex]))

svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 6)))
svmRModel <- train(BHPredictorsTrain[-chasIndex],                  BHOutcomeTrain$medvCat,
                   method = "svmRadial",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)

svmRModel
plot(svmRModel)


# predict on testing set
svmRPred <- predict(svmRModel, BHPredictorsTrain[-chasIndex])
svmRValues <- postResample(pred = svmRPred, obs = BHOutcomeTrain$medvCat) 
svmRValues

#test set
confusionMatrix(data = svmRPred,
                reference = BHPredictorsTrain[-chasIndex])

#train set
confusionMatrix(data = svmRModel$pred$pred,
                reference = svmRModel$pred$obs)

# AUC
svmRPred <- predict(svmRModel, BHPredictorsTrain[-chasIndex], type = "prob")
svmRRoc <- roc(response = svmRModel$pred$obs,
               predictor = svmRModel$pred$high,
               levels = rev(levels(svmRModel$pred$obs)))
plot(svmRRoc, legacy.axes = TRUE)
svmRAUC <- auc(svmRRoc)
svmRAUC





########## Naive Bayes ##########

install.packages("klaR")
library(klaR)
nbFit <- train(BHPredictorsTrain[-chasIndex],                  BHOutcomeTrain$medvCat,
               method = "nb",
               metric = "ROC",
               ## preProc = c("center", "scale"),
               tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
               trControl = ctrl)

nbFit
## plot(nbFit) No tuning parameter for nb

plot(nbFit)


# predict on testing set
nbPred <- predict(nbFit, BHPredictorsTrain[-chasIndex])
nbValues <- postResample(pred = nbPred, obs = BHOutcomeTrain$medvCat) 
nbValues

#test set
confusionMatrix(data = nbPred,
                reference = BHPredictorsTrain[-chasIndex])

#train set
confusionMatrix(data = nbFit$pred$pred,
                reference = nbFit$pred$obs)

# AUC
nbPred <- predict(nbFit, BHPredictorsTrain[-chasIndex], type = "prob")
nbRoc <- roc(response = injuryTrain,
             predictor = nbPred[,1],
             levels = levels(injuryTrain))
plot(nbRoc, legacy.axes = TRUE)
nbAUC <- auc(nbRoc)
nbAUC

#########################################################

# compare testing results
table <- rbind(mdaValues,
               nnetValues,
               fdaValues,
               svmRValues#,nbValues)
)
table

table <- rbind(mdaAUC,
               nnetAUC, 
               fdaAUC,
               svmRAUC)
table


