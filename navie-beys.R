library(readxl)
library(tm)
library(wordcloud)
library(e1071)
library(gmodels)

tweet <- read.csv(file.choose(), header = T,fill = TRUE, stringsAsFactors = F)
tweet$sarcasm <- factor(tweet$sarcasm)
table(tweet$sarcasm)

tweet_corpus <- VCorpus(VectorSource(tweet$text))

tweet_dtm <- DocumentTermMatrix(tweet_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

# creating training and test datasets
tweet_dtm_train <- tweet_dtm[1:1593, ]
tweet_dtm_test  <- tweet_dtm[1594:1992, ]

# also save the labels
tweet_train_labels <- tweet[1:1593, ]$sarcasm
tweet_test_labels  <- tweet[1594:1992, ]$sarcasm

# check that the proportion of spam is similar
prop.table(table(tweet_train_labels))
prop.table(table(tweet_test_labels))

##proportion is not same on train and test data 


rm(tweet_dtm_train)
rm(tweet_dtm_test)
rm(tweet_train_labels)
rm(tweet_test_labels)

# Create random samples
set.seed(123)
train_index <- sample(1000, 800)

tweet_train <- tweet[train_index, ]
tweet_test  <- tweet[-train_index, ]

# check the proportion of class variable
prop.table(table(tweet_train$sarcasm))
prop.table(table(tweet_test$sarcasm))

train_corpus <- VCorpus(VectorSource(tweet_train$text))
test_corpus <- VCorpus(VectorSource(tweet_test$text))

# subset the training data into spam and ham groups
positive <- subset(tweet_train, sarcasm == 1)
negative  <- subset(tweet_train, sarcasm == 0)

wordcloud(positive$text, max.words = 40, scale = c(5, 0.5))
wordcloud(negative$text, max.words = 40, scale = c(5, 0.5))

# create a document-term sparse matrix directly for train and test
train_dtm <- DocumentTermMatrix(train_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

test_dtm <- DocumentTermMatrix(test_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

train_dtm
test_dtm

# create function to convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() convert_counts() to columns of train/test data
train_dtm_binary <- apply(train_dtm, MARGIN = 2, convert_counts)
test_dtm_binary  <- apply(test_dtm, MARGIN = 2, convert_counts)

tweet_classifier <- naiveBayes(as.matrix(train_dtm_binary), tweet_train$sarcasm)

tweet_test_pred <- predict(tweet_classifier, as.matrix(test_dtm_binary))
head(tweet_test_pred)

CrossTable(tweet_test_pred, tweet_test$sarcasm,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
#accuracy is 0.91

#improving accuracy  
tweet_classifier2 <- naiveBayes(as.matrix(train_dtm_binary), tweet_train$sarcasm, laplace = 1)

tweet_test_pred2 <- predict(tweet_classifier2, as.matrix(test_dtm_binary))

CrossTable(tweet_test_pred2, tweet_test$sarcasm,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
#accuracy is 0.92


#improving accuracy  
tweet_classifier3 <- naiveBayes(as.matrix(train_dtm_binary), tweet_train$sarcasm, laplace = .5)

tweet_test_pred3 <- predict(tweet_classifier3, as.matrix(test_dtm_binary))

CrossTable(tweet_test_pred3, tweet_test$sarcasm,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
#accuracy 0.91


write.table(positive$text, file = "E:/Project 3th sem/positive.txt")
write.table(negative$text, file = "E:/Project 3th sem/negative.txt")
