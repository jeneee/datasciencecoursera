library(dplyr)
library(reshape2)
library(ggplot2)
library(caret)

# Load data
training <- read.csv("C:/r/Machear learning/Proyecto a enviar/pml-training.csv", stringsAsFactors = FALSE)%>% tbl_df
testing <- read.csv("C:/r/Machear learning/Proyecto a enviar/pml-testing.csv", stringsAsFactors = FALSE) %>% tbl_df
training$set <- "train"
testing$set <- "test"
classe <- training$classe
problem_id <- testing$problem_id
testing$problem_id <- NULL
a <-rbind(select(training, -classe),testing)
rm(training, testing)
naCols <- apply(a, 2, function(x) sum(is.na(x)))
a <- a[-which(naCols > 0)]
a <- a[-(1:7)]
train <- a[a$set == "train",]
test <- a[a$set == "test",]

train$classe <- classe
test$problem_id <- problem_id
rm(a)

index <- createDataPartition(train$classe, p = 0.8, list = FALSE)

train <- train[index,]
valid <- train[-index,]


nsv <- nearZeroVar(train)
train <- train[,-nsv]
valid <- valid[,-nsv]
test <- test[,-nsv]
scale_center <- preProcess(select(train, -classe), 
                           method = c("scale", "center"))

train_scaled <- predict(scale_center, select(train, -classe))
train_scaled$classe <- train$classe
valid_scaled <- predict(scale_center, select(valid, -classe))
valid_scaled$classe <- valid$classe

test_scaled <- predict(scale_center, select(test, -problem_id))
test_scaled$classe <- test$problem_id

train_scaled$classe <- factor(train_scaled$classe)
valid_scaled$classe <- factor(valid_scaled$classe)

trc <- trainControl("repeatedcv", number = 10, repeats = 2)
gbmGrid <- expand.grid(n.trees = seq(50, 500, 50), 
                       interaction.depth = 1:5,
                       shrinkage = c(0.1, 0.01))

gbm <- train(classe ~ ., data = train_scaled, method = "gbm",  trControl = trc, tuneGrid = gbmGrid)

confusionMatrix(predict(gbm, valid_scaled), valid_scaled$classe)

answers <- predict(gbm, test_scaled)
