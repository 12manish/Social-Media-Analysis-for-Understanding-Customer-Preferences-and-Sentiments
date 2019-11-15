
#Case study B(1)Develop a dictionary-based sentiment analytics engine based on the R library
##'syuzhet' to analyse the different emotions from hotel review tweets

library(syuzhet)
library(tm)
library(ggplot2)
library(dplyr)
setwd("E:\\office")
tweet_data=read.csv("hotel_tweets.csv")

data1<-data.frame(iconv(tweet_data$Negative,from = "latin1", to = "ASCII//TRANSLIT"))
data2<-data.frame(iconv(tweet_data$Positive,from = "latin1", to = "ASCII//TRANSLIT"))

names(data1)="reviews"
names(data2)="reviews"

data=rbind(data1,data2)
dim(data)
str(data)
data$reviews=as.character(newdata$reviews)
reviews.corpus<-Corpus(VectorSource(data$reviews))
summary(reviews.corpus)
inspect(reviews.corpus[1:10])


#Data Transformations -Cleaning

reviews.corpus<-tm_map(reviews.corpus,tolower) #Converting to lower case
reviews.corpus<-tm_map(reviews.corpus,stripWhitespace) #Removing extra white space
reviews.corpus<-tm_map(reviews.corpus,removePunctuation) #Removing punctuations
reviews.corpus<-tm_map(reviews.corpus,removeNumbers) #Removing numbers
my_stopwords<-c(stopwords('english'),"#$<>?+") #Can add more words apart from standard list
reviews.corpus<-tm_map(reviews.corpus,removeWords,my_stopwords)

removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
reviews.corpus <- tm_map(reviews.corpus, content_transformer(removeNumPunct))
## Now using the function 'get_nrc_sentiment'
Emotions <- get_nrc_sentiment(as.character(reviews.corpus))
Emo_res<-data.frame(t(Emotions))
Emotion_df <- data.frame(rowSums(Emo_res))

names(Emotion_df)[1] <- "count"
Emotion_df <- cbind("Emotions" = rownames(Emotion_df), Emotion_df)
rownames(Emotion_df) <- NULL
## ploting a chart to visualise emotions
qplot(Emotions, data=Emotion_df[1:8,], weight=count, geom="bar",fill=Emotions)+
  ggtitle("Hotel tweets Emotions")

# Case study B(2)Develop a machine learning-based model using the R libraries 'tm' and 'e1071' as
##well as evaluate the predictive accuracies of SVM classifier

library(tm)
library(ggplot2)
library(dplyr)
library(e1071)

library(tm)

hotel_tweets_data=read.csv("hotel_tweets.csv")

neg_tweets<-data.frame(iconv(hotel_tweets_data$Negative))
pos_tweets<-data.frame(iconv(hotel_tweets_data$Positive))

names(neg_tweets)="tweets"
names(pos_tweets)="tweets"
neg_tweets$sentiment="negative"
pos_tweets$sentiment="positive"

##Use the first 200 negative tweets and the first 200 positive tweets as the training
##dataset; and use the rest of the 63 negative tweets and 63 positive tweets as the testing dataset


train_pos_tweets=pos_tweets[1:200,]
train_neg_teeets=neg_tweets[1:200,]
test_pos_tweets=pos_tweets[201:263,]
test_neg_tweets=neg_tweets[201:263,]


train_tweets=rbind(train_pos_tweets,train_neg_teeets)
test_tweets=rbind(test_pos_tweets,test_neg_tweets)

train_tweets$type="train"
test_tweets$type="test"
Tweets=rbind(train_tweets,test_tweets)


row.names(Tweets)<-1:nrow(Tweets)
Tweets.corpus=Corpus(VectorSource(Tweets$tweets))

Tweets.corpus<-tm_map(Tweets.corpus,tolower) #Converting to lower case
Tweets.corpus<-tm_map(Tweets.corpus,stripWhitespace) #Removing extra white space
Tweets.corpus<-tm_map(Tweets.corpus,removePunctuation) #Removing punctuations
Tweets.corpus<-tm_map(Tweets.corpus,removeNumbers) #Removing numbers
my_stopwords<-c(stopwords('english'),"#$<>?+") #Can add more words apart from standard list
Tweets.corpus<-tm_map(Tweets.corpus,removeWords,my_stopwords)

Tweets.tdm<- DocumentTermMatrix(Tweets.corpus)
dim(Tweets.tdm) #Dimensions of term document matrix
inspect(Tweets.tdm[1:10,1:10]) #Inspecting the term document matrix
Tweets.spm=removeSparseTerms(Tweets.tdm,0.95)
Tweets.mat=as.matrix(Tweets.spm)

cont = create_container(Tweets.mat, as.numeric(as.factor(Tweets[,2])),
                        trainSize=1:400, testSize=401:526,virgin=FALSE)


## Develop a machine learning-based sentiment analytics engine and predict
##sentiment categories (only 'positive' and 'negative') using 'tm' and 'e1071' with the SVM classifier


mod = train_models(cont, algorithms=c("SVM"))
prediction = classify_models(cont, mod)
class(prediction)
head(prediction)

##Evaluate the testing accuracies and report the predicted results
recall_accuracy(as.numeric(as.factor(test_tweets[,2])), prediction[,"SVM_LABEL"])
library(caret)
test_tweets$sentiment=ifelse(test_tweets$sentiment=="negative",1,2)
test_tweets$sentiment=as.factor(test_tweets$sentiment)

confusionMatrix(test_tweets$sentiment, prediction[,"SVM_LABEL"])


