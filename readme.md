### Introduction

This report uses a [Weight Lifting Exercises Dataset](http://groupware.les.inf.puc-rio.br/har#ixzz3DTR1Lq9c) to build a prediction model for whether or not a person is correctly performing an exercise, and what type of error he or she may be making. Specifically, 

>"[s]ix young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell
>Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the
>elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only
>halfway (Class D) and throwing the hips to the front (Class E)."

The dataset contains readings from a number of sensors attached to the participants while performing the exercises. This report explains how a statistical prediction model was built using these variables in order to predict the type of error performed by the participants. 

### Data cleaning and preparation

As a first step in the analysis, we load and clean the data. About two thirds of the variables in the dataset contain mostly missing values, which were removed. A number of variables relating to the timing of the exercises were also removed, since the documentation on how to interpret these varibles was incomplete.

Although the dataset was already divided into a training and a testing set, the training set was deemed large enough to be split into training and a validation sets (using an 80/20 split). Creating a validation set allowed me to get a more accurate estimate of the test set error rate than I would using only cross-validation on the original training set.

As a final step in the preparation, all variables were scaled and centered, primarily to allow for easier plotting, but also since some ML algorithms need/prefer the data to be thus transformed.




```r
# Load packages
# library(dplyr)
# library(ggplot2)
library(caret)
library(doParallel)

# Register cores for parallel processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Load data
training <- read.csv("./data/pml-training.csv", stringsAsFactors = FALSE) %>% tbl_df
testing <- read.csv("./data/pml-testing.csv", stringsAsFactors = FALSE) %>% tbl_df

training$set <- "train"
testing$set <- "test"
classe <- training$classe
problem_id <- testing$problem_id
testing$problem_id <- NULL

full <- rbind(select(training, -classe), testing)

# Remove variables with mostly NAs
naCols <- apply(full, 2, function(x) sum(is.na(x)))
full <- full[-which(naCols > 0)]

# Remove variables not related to instrument readings
full <- full[-c(1, 3:7)]

# Re-split into training and testing sets
train <- full[full$set == "train",]
test <- full[full$set == "test",]

train$classe <- classe
test$problem_id <- problem_id
rm(full)

# Create training and validation set
set.seed(1)
index <- createDataPartition(train$classe, p = 0.8, list = FALSE)

train2 <- train[index,]
valid <- train[-index,]

# Remove variables with near-zero variance
nzv <- nearZeroVar(train2)
train2 <- train2[,-nzv]
valid <- valid[,-nzv]
test <- test[,-nzv]

# Scale and center the data
scale_center <- preProcess(select(train2, -classe, -user_name), 
                           method = c("scale", "center"))

train_scaled <- predict(scale_center, select(train2, -classe, -user_name))
train_scaled$classe <- train2$classe

valid_scaled <- predict(scale_center, select(valid, -classe, -user_name))
valid_scaled$classe <- valid$classe

train_scaled$classe <- factor(train_scaled$classe)
valid_scaled$classe <- factor(valid_scaled$classe)
train_scaled$user_name <- factor(train2$user_name)
valid_scaled$user_name <- factor(valid$user_name)
```

### Exploratory data analysis 

To get a feel for what the data looked like, a large number of graphs were produced. The first figure below, for example, gives a first indication that the different classes are quite well-separated (we should thus expect a relatively high accuracy rate). 


```r
qplot(data=train, roll_belt, pitch_belt, facets=user_name~classe)
```

![plot of chunk qplot](./readme_files/figure-html/qplot.png) 

The next figure illustrates the high correlation between the readings of different sensors (e.g. high correlation between the sensors for different directions in the movement of the forearm). This suggests that there is probably a high level of redundancy in the predictors, and that a dimension reduction method (e.g. principal components analysis) might work well in trimming down the number of predictors and thus make the model more easily interpretable. However, since we are strictly interested in the prediction accuracy of the model here, no such dimension reduction was performed.


```r
library(corrplot)
train2[-c(1,54)] %>% 
  cor %>% 
  corrplot(method = "color", tl.cex = 0.5)
```

![plot of chunk corrplot](./readme_files/figure-html/corrplot.png) 

### Training a model

A large number of standard ML models were fit to the data. However, due to computational considerations, only the most promising type of model -- the stochastic gradient boosting model (gbm) -- is documented here. The other models tested included linear discriminant analysis, a support vector machine with radial kernel, recursive partitioning tree, and random forests. Using the standard tuning length in the ```caret``` package, they all had CV-estimated accuracies in the .7-.8 range. 

The initial plan was to tune the gbm model using all three tuning parameters (number of trees, the interaction depth, and the shrinkage factor), and evaluate each model using 10-fold cross-validation. However, doing this in only one function call proved to be too computationally intense, for which reason I trained the model using  multiple smaller tuning grids. The code below shows the final function calls. 


```r
# Train models
trc <- trainControl("cv", number = 2)

gbmGrid <- expand.grid(n.trees = seq(300, 500, 50), 
                        interaction.depth = 3:4,
                        shrinkage = c(0.1))

# gbm_valid <- train(y = train_scaled[,53], x = train_scaled[-53], 
#                  method = "gbm", 
#                  trControl = trc, 
#                  tuneGrid = gbmGrid, 
#                  verbose = TRUE)

# save(gbm_valid, file = "gbm_valid.RData")
load(file = "gbm_valid.RData")
gbm_valid
```

```
## Stochastic Gradient Boosting 
## 
## 15699 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (2 fold) 
## 
## Summary of sample sizes: 7849, 7850 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy  Kappa  Accuracy SD  Kappa SD
##   3                  300      1         1      2e-03        3e-03   
##   3                  350      1         1      2e-03        3e-03   
##   3                  400      1         1      1e-03        1e-03   
##   3                  450      1         1      3e-04        3e-04   
##   3                  500      1         1      9e-05        1e-04   
##   4                  300      1         1      9e-05        1e-04   
##   4                  350      1         1      5e-04        7e-04   
##   4                  400      1         1      9e-04        1e-03   
##   4                  450      1         1      1e-03        2e-03   
##   4                  500      1         1      4e-04        5e-04   
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 500,
##  interaction.depth = 4 and shrinkage = 0.1.
```

The best model had a CV-estimated accuracy of 0.987 and the optimal tuning parameters were chosen as 500 trees, an interaction depth of 4, and a shrinkage parameter of 0.1.

### Evaluating the model

To estimate the out-of-sample error rate of the model, predictions for the validation set for calculated and compared against the ground truth in a confusion matrix.


```r
(confMat <- confusionMatrix(predict(gbm_valid, valid_scaled[-53]), valid_scaled$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114   10    0    2    0
##          B    1  742    9    0    0
##          C    0    1  647   25    0
##          D    1    6   20  616    3
##          E    0    0    8    0  718
## 
## Overall Statistics
##                                         
##                Accuracy : 0.978         
##                  95% CI : (0.973, 0.982)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.972         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.978    0.946    0.958    0.996
## Specificity             0.996    0.997    0.992    0.991    0.998
## Pos Pred Value          0.989    0.987    0.961    0.954    0.989
## Neg Pred Value          0.999    0.995    0.989    0.992    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.189    0.165    0.157    0.183
## Detection Prevalence    0.287    0.192    0.172    0.165    0.185
## Balanced Accuracy       0.997    0.987    0.969    0.974    0.997
```

The confusion matrix suggests that classes C and D were the most problematic to predict. Still, the accuracy rate of 
0.978 was promising. 

### Making predictions

Finally, a model identical to the one found above was fit to the entire training set (i.e. including the 3000 or so observations in the validation set not used to fit the previous model). The model was then used to predict the 20 classes in the original training set. 


```r
scale_center_final <- preProcess(select(train, -classe, -user_name, -set), 
                           method = c("scale", "center"))
train_scaled_final <- predict(scale_center_final, select(train, -classe, -user_name, -set))
train_scaled_final$classe <- train$classe
train_scaled_final$user_name <- factor(train$user_name)

test_scaled <- predict(scale_center_final, select(test, -problem_id, -user_name))
test_scaled$problem_id <- test$problem_id
test_scaled$user_name <- factor(test$user_name)
```


```r
# Train model using optimal tuning params from cross-validated earlier model
trc_final <- trainControl("none")

gbmGrid_final <- data.frame(n.trees = gbm_valid$finalModel$n.trees, 
                        interaction.depth = gbm_valid$finalModel$interaction.depth,
                        shrinkage = gbm_valid$finalModel$shrinkage)

# gbm_final <- train(y = train_scaled_final$classe, x = train_scaled_final[-53], 
#              method = "gbm", 
#              trControl = trc_final, 
#              tuneGrid = gbmGrid_final, 
#              verbose = TRUE)

# save(gbm_final, file = "gbm_final.RData")
load("gbm_final.RData")
gbm_final
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    53 predictor
## 
## No pre-processing
## Resampling: None
```


```r
# Function for creating files to submit answers
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
      filename = paste0("./answers/problem_id_",i,".txt")
      write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

# Predicted answers
(answers <- predict(gbm_final, select(test_scaled, -problem_id)))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# pml_write_files(answers)
```

### Conclusion

This report has explained how a machine learning algorithm called "Gradient boosting machine" was fitted to a dataset on sensor-readings in order to predict the type of physical exercise being performed. The model achieved a validation set accuracy rate close to 0.98 and correctly predicted 20/20 observations in a test set.
