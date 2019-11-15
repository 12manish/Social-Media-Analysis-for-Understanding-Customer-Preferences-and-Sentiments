
#############online_news_popularity###########################

library(ggplot2)
library(dplyr)
library(xlsx)
install.packages("readxl")
library("readxl")
setwd("E:\\Office")
## Import the data set
online <- read_excel("online_popularity_data.xlsx")

View(online)
str(online)
summary(online)

### top 5 lower and top 5 higest and unique, mean,
#all percentile show,n, missing also
install.packages("Hmisc")
library(Hmisc)
describe(online)

# for check data frequency of na
install.packages("questionr")
library(questionr)
freq.na(online)

# Check for outliers 

hist(online$shares)

?boxplot
x <- boxplot(online$shares)
x
out <- x$out
out
# Remove Outliers from the shares

index <- which(online$shares %in% x$out)
index
onlineout <- online[-index,]
dim(onlineout)

install.packages("DataExplorer")
library(DataExplorer)
plot_missing(onlineout) 
plot_bar(onlineout) 
plot_histogram(onlineout) 

# Looking at the number of missing values in each column
sapply(onlineout, function(x) sum(is.na(x)) )

# Plot lat and long with shares
ggplot(data = onlineout, aes(y=shares, x=n_tokens_content)) + 
  geom_point(aes(colour=n_tokens_title)) + 
  scale_colour_gradient(low = "blue", high = "white") +
  theme_bw()


#  Count of images and videos
ggplot(onlineout, aes(x=num_imgs, y=num_videos)) + 
  geom_point(aes(colour=n_tokens_title)) +
  theme_bw() + xlab("images count") + 
  ylab("videos")

# Plot of car and bathroom
ggplot(onlineout, aes(x=timedelta, y=n_tokens_title)) + 
  geom_count(aes(colour=as.numeric(as.character(shares)))) +
  scale_colour_gradient(low = "blue", high = "green") +
  theme_bw() + xlab("timedelta") + 
  ylab("n_tokens_title") + labs(color = "num_videos")

# Plot of bathroom and rooms
ggplot(onlineout,aes(x=num_imgs, y=num_videos)) + 
  geom_count(colour = "steelblue") + 
  stat_smooth(method = "lm", col = "red") +
  theme_bw() + xlab("num_imgs") + 
  ylab("num_videos")

# Fit linear model Bathroom ~ Rooms
onlinelm = lm(num_videos ~ num_imgs, 
                 data = onlineout)
summary(onlinelm)

str(onlineout)
# Plot of weekend vs Shares
ggplot(onlineout, aes(x=is_weekend, y=shares)) +
  geom_violin(fill="steelblue", draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw() + theme(axis.text.x = element_text(hjust = 1)) + xlab("is_weekend") + 
  ylab("shares")
# Plot of weekday_is_monday vs Shares
ggplot(onlineout, aes(x=weekday_is_monday, y=shares)) +
  geom_violin(fill="steelblue", draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw() + theme(axis.text.x = element_text(hjust = 1)) + xlab("weekday_is_monday") + 
  ylab("shares")
# Plot of weekday_is_sunday vs Shares
ggplot(onlineout, aes(x=weekday_is_sunday, y=shares)) +
  geom_violin(fill="steelblue", draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw() + theme(axis.text.x = element_text(hjust = 1)) + xlab("weekday_is_sunday") + 
  ylab("shares")
# Plot of weekday_is_saturday vs Shares
ggplot(onlineout, aes(x=weekday_is_saturday, y=shares)) +
  geom_violin(fill="steelblue", draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw() + theme(axis.text.x = element_text(hjust = 1)) + xlab("weekday_is_saturday") + 
  ylab("shares")


# Save the data
write.csv(onlineout, "onnline.csv")

data = read.csv("onnline.csv")
# deciding number of clusters
ggplot(data, aes(shares))+
  geom_histogram(bins=7, color="darkblue", fill="lightblue", size=.1) +
  labs(title="Histogram with 7 Bins") + theme_bw()
ggplot(data, aes(log(shares))) +
  geom_histogram(bins=7, color="darkblue", fill="lightblue", size=.1) +
  labs(title="Histogram with 7 Bins") + theme_bw()

#devide categorical and continous variable variable
data_num=data[,sapply(data, is.numeric)]
data_cat=data[,!sapply(data, is.numeric)]
dim(data_num)
dim(data_cat)

#Remove Na  
is.na(data)
#install.packages("Hmice")                        
                             
library(Hmisc)   

impute(data$Topics, mean)  # replace with mean
impute(data$Content, median)
impute(data$Title, median)
sum(is.na(data))

## Frequency table with table() function in R
##.	Explore the dataset to understand and manage the six types of 
#data channels (lifestyle, entertainment, bus, socmed, tech, world) and the associating data. 
#In each data channel column, the value of 1 represents that the data in the row is of the corresponding data channel.

colnames(data)
table(data$data_channel_is_lifestyle)
table(data$data_channel_is_entertainment)
table(data$data_channel_is_bus)
table(data$data_channel_is_socmed)
table(data$data_channel_is_tech)
table(data$data_channel_is_world)

###Explore the data and investigate what properties of the article correlate with the high number
##of shares of the article on social media.

#Are any of the independent variables correlated?  #MULTICOLLINEARITY IF IT IS THERE DROP

install.packages("corrplot", dependencies = T)
library(corrplot)
#findCorrelation
install.packages("caret")
library(caret)
Cor1=cor(data_num)
findCorrelation(Cor1,cutoff = 0.7,names= TRUE)  #Correlation Coefficient Calculator
#new file after removing correlated values
data_new=data_num[,-c(1,6,8,7)]

#make new file after removing correlated values and combining with 
data_full=cbind(data_new,data_cat)

##Copy the separate datasets for each channel to different Excel sheets
##(sort and filter by each data channel to separate).
##Copy the separate datasets for each channel to different Excel sheets
write.csv(data_full, "on.csv")

m = read.csv("on.csv")

##devide categorical and continous variable variable
dat_num=m[,sapply(m, is.numeric)]
dat_cat=m[,!sapply(m, is.numeric)]
dim(dat_num)
dim(dat_cat)
#Arranging data 
colnames(dat_num)

dat_num1<-select(dat_num, -c(1))
dat_num1<-arrange(dat_num,shares)
dat_num2<-arrange(dat_num1,timedelta)
View(dat_num2)
## filterd shares
m %>% 
  select(shares,timedelta,num_imgs,num_videos)%>%
  filter(shares>=5000)
View(m)


### .	Investigate the following properties and explain how they could have affected the high number of shares. You should provide the explanation to support your argument.
#o	Number of tokens in the title
#o	Number of tokens in the content
#o	Was the article published on the weekend
#o	Number of links
#o	Number of images
#o	Number of videos

table(m$data_channel_is_lifestyle)
table(m$data_channel_is_entertainment)
table(m$data_channel_is_bus)
table(m$data_channel_is_socmed)
table(m$data_channel_is_tech)
table(m$data_channel_is_world)

##To do this, you can create plots in R between the corresponding columns
##and the number of shares. You may want to include a fitted line to your plots to investigate
##the correlation for continuous variables
#Finding correlation between numeric variables 

str(train)
corrln<-cor(dat_num[,c(2:25)])
install.packages("corrgram")
library(corrgram)
?corrgram
cormat<-corrgram(corrln)

write.csv(cormat,"Correlation.csv")
# Plot lat and long with shares
ggplot(data = onlineout, aes(y=shares, x=n_tokens_content)) + 
  geom_point(aes(colour=n_tokens_title)) + 
  scale_colour_gradient(low = "blue", high = "white") +
  theme_bw()


#  Count of images and videos
ggplot(onlineout, aes(x=num_imgs, y=num_videos)) + 
  geom_point(aes(colour=n_tokens_title)) +
  theme_bw() + xlab("images count") + 
  ylab("videos")

# Plot of car and bathroom
ggplot(onlineout, aes(x=timedelta, y=n_tokens_title)) + 
  geom_count(aes(colour=as.numeric(as.character(shares)))) +
  scale_colour_gradient(low = "blue", high = "green") +
  theme_bw() + xlab("timedelta") + 
  ylab("n_tokens_title") + labs(color = "num_videos")


str(onlineout)
# Plot of weekend vs Shares
ggplot(onlineout, aes(x=is_weekend, y=shares)) +
  geom_violin(fill="steelblue", draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw() + theme(axis.text.x = element_text(hjust = 1)) + xlab("is_weekend") + 
  ylab("shares")
# Plot of weekday_is_monday vs Shares
ggplot(onlineout, aes(x=weekday_is_monday, y=shares)) +
  geom_violin(fill="steelblue", draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw() + theme(axis.text.x = element_text(hjust = 1)) + xlab("weekday_is_monday") + 
  ylab("shares")
# Plot of weekday_is_sunday vs Shares
ggplot(onlineout, aes(x=weekday_is_sunday, y=shares)) +
  geom_violin(fill="steelblue", draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw() + theme(axis.text.x = element_text(hjust = 1)) + xlab("weekday_is_sunday") + 
  ylab("shares")
# Plot of weekday_is_saturday vs Shares
ggplot(onlineout, aes(x=weekday_is_saturday, y=shares)) +
  geom_violin(fill="steelblue", draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw() + theme(axis.text.x = element_text(hjust = 1)) + xlab("weekday_is_saturday") + 
  ylab("shares")

ggplot(data, aes(shares))+
  geom_histogram(bins=7, color="darkblue", fill="lightblue", size=.1) +
  labs(title="Histogram with 7 Bins") + theme_bw()
ggplot(data, aes(log(shares))) +
  geom_histogram(bins=7, color="darkblue", fill="lightblue", size=.1) +
  labs(title="Histogram with 7 Bins") + theme_bw()


### fitted line to your plots to investigate the correlation for continuous variables
ggplot(onlineout,aes(x=num_imgs, y=num_videos)) + 
  geom_count(colour = "steelblue") + 
  stat_smooth(method = "lm", col = "red") +
  theme_bw() + xlab("num_imgs") + 
  ylab("num_videos")

ggplot(onlineout,aes(x=shares, y=num_hrefs)) + 
  geom_count(colour = "steelblue") + 
  stat_smooth(method = "lm", col = "red") +
  theme_bw() + xlab("shares") + 
  ylab("num_hrefs")

###################END####################################################################

