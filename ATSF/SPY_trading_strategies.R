rm(list=ls())


inst_pack<-rownames(installed.packages())

if (!"neuralnet"%in%inst_pack)
  install.packages("neuralnet")
if (!"fGarch"%in%inst_pack)
  install.packages("fGarch")
if (!"xts"%in%inst_pack)
  install.packages("xts")
if (!"xts"%in%inst_pack)
  install.packages("quantmod")


# Installiere und lade benötigte Pakete
if (!require(quantmod)) install.packages("quantmod", dependencies = TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)

library(neuralnet)
library(fGarch)
library(xts)
library(quantmod)

# Lade die Pakete
library(quantmod)
library(ggplot2)
library(dplyr)

#################################################################################
# path <- "C:/Users/Oscar/OneDrive - ZHAW/s - FS 2025/ATSF/Project/ATSP_project"
# setwd(path)
# getwd()
########################################################################
# ---- 1️⃣ Lade SPY-Daten von Yahoo Finance ----
symbol <- "SPY"
getSymbols(symbol, src = "yahoo", from = "1993-01-01", to = Sys.Date(), auto.assign = TRUE)
df <- get(symbol)
head(df)
# Erstelle Dataframe mit relevanten Spalten
df <- data.frame(Date = index(df),
                 Open = as.numeric(Op(df)),
                 Close = as.numeric(Cl(df)))

# Berechne Renditen für alle drei Strategien
df <- df %>%
  mutate(Overnight_Return = (Open / lag(Close)) - 1,   # Rendite: Buy Close, Sell Next Open
         Intraday_Return = (Close / Open) - 1,         # Rendite: Buy Open, Sell Close
         Daily_Return = (Close / lag(Close)) - 1)      # Rendite: Buy and Hold (tägliche Rendite)

len_before <- dim(df)[1]

# Entferne NA-Werte
df <- na.omit(df)

len_after <- dim(df)[1]

# number of removed rows
len_before - len_after

# Geometrische Kumulation der Renditen
df <- df %>%
  mutate(Overnight_Strategy = 100 * (cumprod(1 + Overnight_Return) - 1),
         Intraday_Strategy = 100 * (cumprod(1 + Intraday_Return) - 1),
         Buy_Hold_Strategy = 100 * (cumprod(1 + Daily_Return) - 1))

# head(df)
#####################################3
# some plots

par(mfrow=c(2,1))
plot(df$Open, col=1, main="Open")
plot(df$Close, col=2, main="Close")

par(mfrow=c(1,1))
ts.plot(diff(log(df$Close)), col=3, main="Log-returns")


###############################################
# plots in 'final design'

# # ---- 2️⃣ Erstelle SPY-Plot im exakten Stil ----
# plot_spy <- ggplot(df, aes(x = Date)) +
#   geom_line(aes(y = Overnight_Strategy, color = "Buy Close, Sell Next Open"), linewidth = 1.2) +
#   geom_line(aes(y = Intraday_Strategy, color = "Buy Open, Sell Close"), linewidth = 1.2) +
#   geom_line(aes(y = Buy_Hold_Strategy, color = "Buy and Hold"), linewidth = 1.2) +
#   scale_color_manual(values = c("Buy Close, Sell Next Open" = "green", 
#                                 "Buy Open, Sell Close" = "red",
#                                 "Buy and Hold" = "blue")) +
#   scale_y_continuous(breaks = seq(0, max(df$Overnight_Strategy, df$Intraday_Strategy, df$Buy_Hold_Strategy, na.rm = TRUE), by = 100)) +  # Y-Achse in 100er-Schritten
#   scale_x_date(date_breaks = "2 years", date_labels = "%Y") +  # Zweijährliche Intervalle für bessere Lesbarkeit
#   labs(title = "S&P 500 (SPY) After Hours vs. Regular Trading vs. Buy-and-Hold Performance",
#        x = "Date",
#        y = "Cumulative Return (%)",
#        color = "") +  # Keine Legendenüberschrift
#   theme_minimal() +
#   theme(legend.position = "top",
#         plot.title = element_text(face = "bold", size = 14),
#         axis.text = element_text(size = 12),
#         axis.title = element_text(size = 13),
#         axis.text.x = element_text(angle = 45, hjust = 1))  # Dreht die X-Achsenbeschriftungen für bessere Lesbarkeit

########################################################################################-----------------------------
# Fit a GARCH to log-returns of 'Close'


x_fit<-as.ts(na.omit(diff(log(df$Close))))
# GARCH(1,1)
y.garch_11<-garchFit(~garch(1,1),data=x_fit,delta=2,include.delta=F,include.mean=F,trace=F)

ts.plot(x_fit)
lines(y.garch_11@sigma.t,col="red")
#-----------------

standard_residuals<-y.garch_11@residuals/y.garch_11@sigma.t
ts.plot(standard_residuals)
#-----------------

par(mfrow=c(2,1))
acf(x_fit,main="Acf log-returns",ylim=c(0,0.1))
acf(standard_residuals,main="Acf standardized residuals GARCH(1,1)",ylim=c(0,0.1))


# Fit a GARCH to log-returns of 'Open'

x_fit<-as.ts(na.omit(diff(log(df$Open))))
# GARCH(1,1)
y.garch_11<-garchFit(~garch(1,1),data=x_fit,delta=2,include.delta=F,include.mean=F,trace=F)
par(mfrow=c(1,1))
ts.plot(x_fit)
lines(y.garch_11@sigma.t,col="red")
#-----------------

standard_residuals<-y.garch_11@residuals/y.garch_11@sigma.t
ts.plot(standard_residuals)
#-----------------

par(mfrow=c(2,1))
acf(x_fit,main="Acf log-returns",ylim=c(0,0.1))
acf(standard_residuals,main="Acf standardized residuals GARCH(1,1)",ylim=c(0,0.1))

# acf and residuals acf sugest a dependency on lag 12

# Specify target and explanatory data: we use first 12 lags based on above data analysis

###########################################################

# convert data to xts

spy_xts <- xts(df[, -1], order.by = df$Date)

###########################################################

head(df)
head(spy_xts)


x<-na.omit(diff(log(spy_xts$Close)))
head(x)
data_mat <- cbind(
  x,
  stats::lag(x, -1),
  stats::lag(x, -2),
  stats::lag(x, -3),
  stats::lag(x, -4),
  stats::lag(x, -5),
  stats::lag(x, -6),
  stats::lag(x, -7),
  stats::lag(x, -8),
  stats::lag(x, -9),
  stats::lag(x, -10),
  stats::lag(x, -11),
  stats::lag(x, -12)
)

head(data_mat)


# Check length of time series before na.exclude
dim(data_mat)
data_mat<-na.exclude(data_mat)
# Check length of time series after removal of NAs
dim(data_mat)
head(data_mat)
tail(data_mat)

# Specify in- and out-of-sample episodes
in_out_sample_separator <- index(data_mat)[round(dim(data_mat)[1]/2)] # "2009-02-20"


target_in<-data_mat[paste("/",in_out_sample_separator,sep=""),1]
tail(target_in)
explanatory_in<-data_mat[paste("/",in_out_sample_separator,sep=""),2:ncol(data_mat)]
tail(explanatory_in)

target_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),1]
head(target_out)
tail(target_out)
explanatory_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),2:ncol(data_mat)]


train<-cbind(target_in,explanatory_in)
test<-cbind(target_out,explanatory_out)
head(test)
tail(test)
nrow(train)
nrow(test)

#-------------------------------------------------------------------------------
#  Neural net fitting

# Scaling data for the NN

maxs <- apply(data_mat, 2, max) 
mins <- apply(data_mat, 2, min)
# Transform data into [0,1]  
scaled <- scale(data_mat, center = mins, scale = maxs - mins)

apply(scaled,2,min)
apply(scaled,2,max)
#-----------------
# 4.b
# Train-test split
train_set <- scaled[paste("/",in_out_sample_separator,sep=""),]
test_set <- scaled[paste(in_out_sample_separator,"/",sep=""),]

train_set<-as.matrix(train_set)
test_set<-as.matrix(test_set)
#-----------------------------------

colnames(train_set)<-paste("lag",0:(ncol(train_set)-1),sep="")
n <- colnames(train_set)
# Model: target is current bitcoin, all other variables are explanatory  
f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))

tail(train_set)

# Set/fix the random seed 
set.seed(3)
nn <- neuralnet(f,data=train_set,hidden=c(3,2),linear.output=F)
#------------------------------------
# 4.d (compare different realizations of the above net: train the net without specifying set.seed)
plot(nn)

# In sample performance
# 1. Without re-scaling: MSE based on transformed data
MSE.in.nn<-mean(((train_set[,1]-nn$net.result[[1]])*(max(data_mat[,1])-min(data_mat[,1])))^2)
MSE.in.nn
# 2. With re-scaling: MSE based on scale of original data
scaling_term<-(max(data_mat[,1])-min(data_mat[,1]))
MSE.in.nn<-mean(((train_set[,1]-nn$net.result[[1]])*scaling_term)^2)
MSE.in.nn

# Out-of-sample performance
# 1. Compute out-of-sample predictions based on transformed data
# Provide test-data to predict: use explanatory columns 2:ncol(test_set) (First column is forecast target)
pr.nn <- predict(nn,test_set[,2:ncol(test_set)])
predicted_scaled<-pr.nn
# Numbers are between 0 and 1
tail(predicted_scaled)
# Transform forecasts back to original data: rescale and shift by min(data_mat[,1])
predicted_nn <- predicted_scaled*scaling_term+min(data_mat[,1])
test.r <- test_set[,1]*scaling_term+min(data_mat[,1])
# Check: test.r is the same as test[,1]
test[,1]-test.r
# Calculating MSE
MSE.out.nn <- sum((test.r - predicted_nn)^2)/nrow(test_set)

# Compare in-sample and out-of-sample
c(MSE.in.nn,MSE.out.nn)
# 
# # Compare Regression and nn in-sample: which model would you prefer/select?
# print(paste(MSE.in.lm,MSE.in.nn))
# # Compare Regression and nn in-sample: which model was better
# print(paste(MSE.out.lm,MSE.out.nn))



#--------------------------------
# 4.f Trading performance
perf_nn<-(sign(predicted_nn))*target_out


sharpe_nn<-sqrt(365)*mean(perf_nn,na.rm=T)/sqrt(var(perf_nn,na.rm=T))
par(mfrow=c(1,1))
plot(cumsum(perf_nn),main=paste("NN cumulated performances out-of-sample, sharpe=",round(sharpe_nn,2),sep=""))

####################################################################


estimate_nn<-function(train_set,number_neurons,data_mat,test_set,f)
{
  nn <- neuralnet(f,data=train_set,hidden=number_neurons,linear.output=T)
  
  
  # In sample performance
  predicted_scaled_in_sample<-nn$net.result[[1]]
  # Scale back from interval [0,1] to original log-returns
  predicted_nn_in_sample<-predicted_scaled_in_sample*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # In-sample MSE
  MSE.in.nn<-mean(((train_set[,1]-predicted_scaled_in_sample)*(max(data_mat[,1])-min(data_mat[,1])))^2)
  
  # Out-of-sample performance
  # Compute out-of-sample forecasts
  pr.nn <- predict(nn,as.matrix(test_set[,2:ncol(test_set)]))
  predicted_scaled<-pr.nn
  # Results from NN are normalized (scaled)
  # Descaling for comparison
  predicted_nn <- predicted_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  test.r <- test_set[,1]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
  MSE.out.nn <- mean((test.r - predicted_nn)^2)
  
  # Compare in-sample and out-of-sample
  MSE_nn<-c(MSE.in.nn,MSE.out.nn)
  return(list(MSE_nn=MSE_nn,predicted_nn=predicted_nn,predicted_nn_in_sample=predicted_nn_in_sample))
  
}

#####################################################################################
#####################################################################################


# overnigth-returns strategy

# Erstelle Dataframe mit relevanten Spalten
symbol <- "SPY"
getSymbols(symbol, src = "yahoo", from = "1993-01-01", to = Sys.Date(), auto.assign = TRUE)
df <- get(symbol)

df <- data.frame(Date = index(df),
                 Open = as.numeric(Op(df)),
                 Close = as.numeric(Cl(df)))

# Berechne Renditen für alle drei Strategien
df <- df %>%
  mutate(Open_log = log(Open),
         Close_log = log(Close)) %>%
  mutate(Overnight_log_Return = (Open_log / lag(Close_log)) - 1,   # Rendite: Buy Close, Sell Next Open
         Intraday_log_Return = (Close_log / Open_log) - 1,         # Rendite: Buy Open, Sell Close
         Daily_log_Return = (Close_log / lag(Close_log)) - 1)      # Rendite: Buy and Hold (tägliche Rendite)

len_before <- dim(df)[1]

# Entferne NA-Werte
df <- na.omit(df)

len_after <- dim(df)[1]

# number of removed rows
len_before - len_after

head(df)

# ...................


# convert data to xts

spy_xts <- xts(df[, -1], order.by = df$Date)
dim(spy_xts)
x<-na.omit(spy_xts$Overnight_log_Return)
head(x)

data_mat <- cbind(
  x,
  stats::lag(x, -1),
  stats::lag(x, -2),
  stats::lag(x, -3),
  stats::lag(x, -4),
  stats::lag(x, -5),
  stats::lag(x, -6),
  stats::lag(x, -7),
  stats::lag(x, -8),
  stats::lag(x, -9),
  stats::lag(x, -10),
  stats::lag(x, -11),
  stats::lag(x, -12)
)

head(data_mat)


# Check length of time series before na.exclude
dim(data_mat)
data_mat<-na.exclude(data_mat)
# Check length of time series after removal of NAs
dim(data_mat)
head(data_mat)
tail(data_mat)

# Specify in- and out-of-sample episodes
in_out_sample_separator <- index(data_mat)[round(dim(data_mat)[1]/2)] # "2009-02-19"


target_in<-data_mat[paste("/",in_out_sample_separator,sep=""),1]
tail(target_in)
explanatory_in<-data_mat[paste("/",in_out_sample_separator,sep=""),2:ncol(data_mat)]
tail(explanatory_in)

target_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),1]
head(target_out)
tail(target_out)
explanatory_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),2:ncol(data_mat)]


train<-cbind(target_in,explanatory_in)
test<-cbind(target_out,explanatory_out)
head(test)
tail(test)
nrow(train)
nrow(test)

#-------------------------------------------------------------------------------
#  Neural net fitting

# Scaling data for the NN

maxs <- apply(data_mat, 2, max) 
mins <- apply(data_mat, 2, min)
# Transform data into [0,1]  
scaled <- scale(data_mat, center = mins, scale = maxs - mins)

apply(scaled,2,min)
apply(scaled,2,max)
#-----------------
# 4.b
# Train-test split
train_set <- scaled[paste("/",in_out_sample_separator,sep=""),]
test_set <- scaled[paste(in_out_sample_separator,"/",sep=""),]

train_set<-as.matrix(train_set)
test_set<-as.matrix(test_set)
#-----------------------------------

colnames(train_set)<-paste("lag",0:(ncol(train_set)-1),sep="")
n <- colnames(train_set)
#  
f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))

tail(train_set)

# Set/fix the random seed 
set.seed(3)
nn <- neuralnet(f,data=train_set,hidden=c(3,2),linear.output=F)
#------------------------------------
# 4.d (compare different realizations of the above net: train the net without specifying set.seed)
plot(nn)

# In sample performance
# 1. Without re-scaling: MSE based on transformed data
MSE.in.nn<-mean(((train_set[,1]-nn$net.result[[1]])*(max(data_mat[,1])-min(data_mat[,1])))^2)
MSE.in.nn
# 2. With re-scaling: MSE based on scale of original data
scaling_term<-(max(data_mat[,1])-min(data_mat[,1]))
MSE.in.nn<-mean(((train_set[,1]-nn$net.result[[1]])*scaling_term)^2)


# Out-of-sample performance
# 1. Compute out-of-sample predictions based on transformed data
# Provide test-data to predict: use explanatory columns 2:ncol(test_set) (First column is forecast target)
pr.nn <- predict(nn,test_set[,2:ncol(test_set)])
predicted_scaled<-pr.nn
# Numbers are between 0 and 1
tail(predicted_scaled)
# Transform forecasts back to original data: rescale and shift by min(data_mat[,1])
predicted_nn <- predicted_scaled*scaling_term+min(data_mat[,1])
test.r <- test_set[,1]*scaling_term+min(data_mat[,1])
# Check: test.r is the same as test[,1]
test[,1]-test.r
# Calculating MSE
MSE.out.nn <- sum((test.r - predicted_nn)^2)/nrow(test_set)

# Compare in-sample and out-of-sample
c(MSE.in.nn,MSE.out.nn)
# 
# # Compare Regression and nn in-sample: which model would you prefer/select?
# print(paste(MSE.in.lm,MSE.in.nn))
# # Compare Regression and nn in-sample: which model was better
# print(paste(MSE.out.lm,MSE.out.nn))



#--------------------------------
# 4.f Trading performance
perf_nn<-(sign(predicted_nn))*target_out

sharpe_nn<-sqrt(365)*mean(perf_nn,na.rm=T)/sqrt(var(perf_nn,na.rm=T))
par(mfrow=c(1,1))
plot(cumsum(perf_nn),main=paste("NN cumulated performances out-of-sample, sharpe=",round(sharpe_nn,2),sep=""))



###############################################################################
#==================================================================================
  
# SPY TRADING STRATEGIES 
# overnigth-returns strategy

# DOWNLOADING THE DATA WILL BE DONE ONLY ONE TIME!
# # Erstelle Dataframe mit relevanten Spalten
# symbol <- "SPY"
# getSymbols(symbol, src = "yahoo", from = "1993-01-01", to = Sys.Date(), auto.assign = TRUE)
# df <- get(symbol)
# 
# df <- data.frame(Date = index(df),
#                  Open = as.numeric(Op(df)),
#                  Close = as.numeric(Cl(df)))
# 
# saveRDS(df, "SPY_data_project.rds")

## Run code from here

df <- readRDS("SPY_data_project.rds")

# Berechne Renditen für alle drei Strategien
df <- df %>%
  mutate(Open_log = log(Open),
         Close_log = log(Close)) %>%
  mutate(Overnight_log_Return = Open_log - lag(Close_log),# Rendite: Buy Close, Sell Next Open
         Intraday_log_Return = Close_log - Open_log,# Rendite: Buy Open, Sell Close
         Daily_log_Return = Close_log - lag(Close_log)) # Rendite: Buy and Hold (tägliche Rendite)

len_before <- dim(df)[1]

# Entferne NA-Werte
df <- na.omit(df)

len_after <- dim(df)[1]

# number of removed rows
len_before - len_after

head(df)



###############################################

# Fit a GARCH to log-returns of 'Overnight_log_Return'


x_fit<-as.ts(na.omit(df$Overnight_log_Return))
# GARCH(1,1)
y.garch_11<-garchFit(~garch(1,1),data=x_fit,delta=2,include.delta=F,include.mean=F,trace=F)

ts.plot(x_fit)
lines(y.garch_11@sigma.t,col="red")
#-----------------

standard_residuals<-y.garch_11@residuals/y.garch_11@sigma.t
ts.plot(standard_residuals)
#-----------------

par(mfrow=c(2,1))
acf(x_fit,main="Acf log-returns",ylim=c(0,0.1))
acf(standard_residuals,main="Acf standardized residuals GARCH(1,1)",ylim=c(0,0.1))


# acf and residuals acf sugest a dependency on lag 13, also on lag 29?

# Specify target and explanatory data: we use first 13 lags based on above data analysis



# convert data to xts

spy_xts <- xts(df[, -1], order.by = df$Date)
dim(spy_xts)
x<-na.omit(spy_xts$Overnight_log_Return)
head(x)

data_mat <- cbind(
  x,
  stats::lag(x, -1),
  stats::lag(x, -2),
  stats::lag(x, -3),
  stats::lag(x, -4),
  stats::lag(x, -5),
  stats::lag(x, -6),
  stats::lag(x, -7),
  stats::lag(x, -8),
  stats::lag(x, -9),
  stats::lag(x, -10),
  stats::lag(x, -11),
  stats::lag(x, -12),
  stats::lag(x, -13)
)

head(data_mat)


# Check length of time series before na.exclude
dim(data_mat)
data_mat<-na.exclude(data_mat)
# Check length of time series after removal of NAs
dim(data_mat)
head(data_mat)
tail(data_mat)

# # Specify in- and out-of-sample episodes
# in_out_sample_separator <- index(data_mat)[round(dim(data_mat)[1]*0.8)] # "2018-10-12"

in_out_sample_separator <- "2018-10-12"

target_in<-data_mat[paste("/",in_out_sample_separator,sep=""),1]
tail(target_in)
explanatory_in<-data_mat[paste("/",in_out_sample_separator,sep=""),2:ncol(data_mat)]
tail(explanatory_in)

target_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),1]
head(target_out)
tail(target_out)
explanatory_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),2:ncol(data_mat)]


train<-cbind(target_in,explanatory_in)
test<-cbind(target_out,explanatory_out)
head(test)
tail(test)
nrow(train)
nrow(test)

#-------------------------------------------
# ## ========================================================================================

# Fitting linear model to log-returns (could be applied to standardized log-returns: returns divided by vola)
lm.fit <- lm(target_in~explanatory_in)#mean(target_in)plot(cumsum(target_in))plot(log(dat$Bid))
summary(lm.fit)
# Without intercept
lm.fit <- lm(target_in~explanatory_in-1)
summary(lm.fit)
#----------------------------------------------
# 
# Predicted data from lm
#   Without intercept
predicted_lm<-explanatory_out%*%lm.fit$coef

# Same as above single line of code (but more explicit: row-wise computation)
for (i in 1:nrow(explanatory_out))
{
  predicted_lm[i,]<-sum(explanatory_out[i,]*lm.fit$coef)
}  

# With intercept: we have to add a column of 1s to the explanatory data (for the additional intercept)
if (length(lm.fit$coef)>ncol(explanatory_out))
  predicted_lm<-cbind(rep(1,nrow(explanatory_out)),explanatory_out)%*%lm.fit$coef

head(predicted_lm)
#------------------------------------------
# 3.h
# Test MSE: in-sample vs. out-of-sample
MSE.in.lm<-mean(lm.fit$residuals^2)
MSE.out.lm <- sum((predicted_lm - target_out)^2)/nrow(test)
c(MSE.in.lm,MSE.out.lm)
#--------------------------------
# 3.i Trading performance
perf_lm<-(sign(predicted_lm))*target_out


sharpe_lm<-sqrt(365)*mean(perf_lm,na.rm=T)/sqrt(var(perf_lm,na.rm=T))
par(mfrow=c(1,1))
plot(cumsum(perf_lm),main=paste("Linear regression cumulated performances out-of-sample, sharpe=",round(sharpe_lm,2),sep=""))

## ========================================================================================
#-------------------------------------------------------------------------------
#  Prepare the datasets for NN

# Scaling data for the NN

maxs <- apply(data_mat, 2, max) 
mins <- apply(data_mat, 2, min)
# Transform data into [0,1]  
scaled <- scale(data_mat, center = mins, scale = maxs - mins)

apply(scaled,2,min)
apply(scaled,2,max)
#-----------------

# Train-test split
train_set <- scaled[paste("/",in_out_sample_separator,sep=""),]
test_set <- scaled[paste(in_out_sample_separator,"/",sep=""),]

train_set<-as.matrix(train_set)
test_set<-as.matrix(test_set)
#-----------------------------------

colnames(train_set)<-paste("lag",0:(ncol(train_set)-1),sep="")
n <- colnames(train_set)
#  
f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))

tail(train_set)


##################################################################################################
# fit a bigg enough NN -  c(20,10)

# Set/fix the random seed 
set.seed(4)
# nn <- neuralnet(f,data=train_set,hidden=c(20,10),linear.output=F) 

# setwd("C:/Users/Oscar/OneDrive - ZHAW/s - FS 2025/ATSF/Project/ATSP_project")

# save(nn, file = "nn_model_20_10.RData")


# load the model
# load("nn_model_20_10.RData")


#------------------------------------
# plot(nn)

# In sample performance
# 1. Without re-scaling: MSE based on transformed data
MSE.in.nn<-mean(((train_set[,1]-nn$net.result[[1]])*(max(data_mat[,1])-min(data_mat[,1])))^2)
MSE.in.nn
# 2. With re-scaling: MSE based on scale of original data
scaling_term<-(max(data_mat[,1])-min(data_mat[,1]))
MSE.in.nn_scaling_term<-mean(((train_set[,1]-nn$net.result[[1]])*scaling_term)^2)


##################################################
# Out-of-sample performance
# 1. Compute out-of-sample predictions based on transformed data
# Provide test-data to predict: use explanatory columns 2:ncol(test_set) (First column is forecast target)
pr.nn <- predict(nn,test_set[,2:ncol(test_set)])
predicted_scaled<-pr.nn
# Numbers are between 0 and 1
tail(predicted_scaled)

# Transform forecasts back to original data: rescale and shift by min(data_mat[,1])
predicted_nn <- predicted_scaled*scaling_term+min(data_mat[,1])

test.r <- test_set[,1]*scaling_term+min(data_mat[,1])
# Check: test.r is the same as test[,1]
# test[,1]-test.r
# Calculating MSE
MSE.out.nn <- sum((test.r - predicted_nn)^2)/nrow(test_set)

# Compare in-sample and out-of-sample
c(MSE.in.nn,MSE.out.nn)

#--------------------------------
# Trading performance
perf_nn<-(sign(predicted_nn))*target_out


sharpe_nn<-sqrt(365)*mean(perf_nn,na.rm=T)/sqrt(var(perf_nn,na.rm=T))
par(mfrow=c(1,1))
plot(cumsum(perf_nn),main=paste("NN cumulated performances out-of-sample,nn(20,10), sharpe=",round(sharpe_nn,2),sep=""))

## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ALTERNATIVE "LAG 29"
# Prepare Data using until lag 29

data_mat_l_29 <- cbind(
  x,
  stats::lag(x, -1),
  stats::lag(x, -2),
  stats::lag(x, -3),
  stats::lag(x, -4),
  stats::lag(x, -5),
  stats::lag(x, -6),
  stats::lag(x, -7),
  stats::lag(x, -8),
  stats::lag(x, -9),
  stats::lag(x, -10),
  stats::lag(x, -11),
  stats::lag(x, -12),
  stats::lag(x, -13),
  stats::lag(x, -14),
  stats::lag(x, -15),
  stats::lag(x, -16),
  stats::lag(x, -17),
  stats::lag(x, -18),
  stats::lag(x, -19),
  stats::lag(x, -20),
  stats::lag(x, -21),
  stats::lag(x, -22),
  stats::lag(x, -23),
  stats::lag(x, -24),
  stats::lag(x, -25),
  stats::lag(x, -26),
  stats::lag(x, -27),
  stats::lag(x, -28),
  stats::lag(x, -29)
)

head(data_mat_l_29)


# Check length of time series before na.exclude
dim(data_mat_l_29)
data_mat_l_29<-na.exclude(data_mat_l_29)
# Check length of time series after removal of NAs
dim(data_mat_l_29)
head(data_mat_l_29)
tail(data_mat_l_29)

# # Specify in- and out-of-sample episodes
# in_out_sample_separator <- index(data_mat)[round(dim(data_mat)[1]*0.8)] # "2018-10-12"

in_out_sample_separator_l_29 <- "2018-10-12"

target_in_l_29<-data_mat_l_29[paste("/",in_out_sample_separator_l_29,sep=""),1]
tail(target_in_l_29)
explanatory_in_l_29<-data_mat_l_29[paste("/",in_out_sample_separator_l_29,sep=""),2:ncol(data_mat_l_29)]
tail(explanatory_in_l_29)

target_out_l_29<-data_mat_l_29[paste(in_out_sample_separator_l_29,"/",sep=""),1]
head(target_out_l_29)
tail(target_out_l_29)
explanatory_out_l_29<-data_mat_l_29[paste(in_out_sample_separator_l_29,"/",sep=""),2:ncol(data_mat_l_29)]


train_l_29<-cbind(target_in_l_29,explanatory_in_l_29)
test_l_29<-cbind(target_out_l_29,explanatory_out_l_29)
head(test_l_29)
tail(test_l_29)
nrow(train_l_29)
nrow(test_l_29)

####################
#  Prepare the datasets for NN

# Scaling data for the NN

maxs <- apply(data_mat_l_29, 2, max) 
mins <- apply(data_mat_l_29, 2, min)
# Transform data into [0,1]  
scaled <- scale(data_mat_l_29, center = mins, scale = maxs - mins)

apply(scaled,2,min)
apply(scaled,2,max)
#-----------------

# Train-test split
train_set_l_29 <- scaled[paste("/",in_out_sample_separator_l_29,sep=""),]
test_set_l_29 <- scaled[paste(in_out_sample_separator_l_29,"/",sep=""),]

train_set_l_29<-as.matrix(train_set_l_29)
test_set_l_29<-as.matrix(test_set_l_29)
#-----------------------------------

colnames(train_set_l_29)<-paste("lag",0:(ncol(train_set_l_29)-1),sep="")
n <- colnames(train_set_l_29)
#  
f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))

tail(train_set_l_29)

###====================================

# fit a bigg enough NN -  c(20,10) for the alternative lag 29
# the first try fitting a c(20,10) was not successful, so we first try a c(12,6)

# Set/fix the random seed 
set.seed(4)
nn <- neuralnet(f,data=train_set_l_29,hidden=c(12,6),linear.output=F) 
# getwd()
# setwd("C:/Users/Oscar/OneDrive - ZHAW/s - FS 2025/ATSF/Project/ATSP_project")

# save(nn, file = "nn_model_12_6_lag_29.RData")


# load the model
# load("nn_model_20_10.RData")


#------------------------------------
# plot(nn)

# In sample performance
# 1. Without re-scaling: MSE based on transformed data
MSE.in.nn<-mean(((train_set_l_29[,1]-nn$net.result[[1]])*(max(data_mat_l_29[,1])-min(data_mat_l_29[,1])))^2)
MSE.in.nn
# 2. With re-scaling: MSE based on scale of original data
scaling_term<-(max(data_mat_l_29[,1])-min(data_mat_l_29[,1]))
MSE.in.nn_scaling_term<-mean(((train_set_l_29[,1]-nn$net.result[[1]])*scaling_term)^2)


##################################################
# Out-of-sample performance
# 1. Compute out-of-sample predictions based on transformed data
# Provide test-data to predict: use explanatory columns 2:ncol(test_set) (First column is forecast target)
pr.nn <- predict(nn,test_set_l_29[,2:ncol(test_set_l_29)])
predicted_scaled<-pr.nn
# Numbers are between 0 and 1
tail(predicted_scaled)

# Transform forecasts back to original data: rescale and shift by min(data_mat[,1])
predicted_nn <- predicted_scaled*scaling_term+min(data_mat_l_29[,1])
head(predicted_nn)
ts.plot(predicted_nn)
test.r_l_19 <- test_set_l_29[,1]*scaling_term+min(data_mat_l_29[,1])
# Check: test.r is the same as test[,1]
# test[,1]-test.r
# Calculating MSE
MSE.out.nn <- sum((test.r_l_19 - predicted_nn)^2)/nrow(test_set_l_29)

# Compare in-sample and out-of-sample
c(MSE.in.nn,MSE.out.nn)

#--------------------------------
# Trading performance
perf_nn<-(sign(predicted_nn))*target_out_l_29


sharpe_nn<-sqrt(365)*mean(perf_nn,na.rm=T)/sqrt(var(perf_nn,na.rm=T))
par(mfrow=c(1,1))
plot(cumsum(perf_nn),main=paste("NN cumulated performances out-of-sample,nn(20,10), sharpe=",round(sharpe_nn,2),sep=""))


# now we try to train a c(20,10) so that we can compare with the previos lag 12

set.seed(4)
nn <- neuralnet(f,data=train_set_l_29,hidden=c(20,10),linear.output=F) # takes less than 1 hour to converge
# save(nn, file = "nn_model_20_10_lag_29.RData")


# 12,6 lag 13 x 10 
# 20,10 lag 13 x 10

getwd()
##############################################################

# Train and save 10 NNs (12,6) using the 13 lag train_set

train_and_save_nn <- function(train_set, number_neurons, f, nn_name) {
  nn <- neuralnet(f, data = train_set, hidden = number_neurons, linear.output = TRUE)
  save(nn, file = paste0(nn_name, ".RData"))
}

num_models <- 10
number_neurons <- c(12, 6)
neurons_str <- paste(number_neurons, collapse = "_")


pb <- txtProgressBar(min = 1, max = num_models, style = 3)

for (i in 1:num_models) {
  nn_name <- paste0("nn_model_lag13__", neurons_str, "_", i)
  train_and_save_nn(train_set, number_neurons, f, nn_name)
  setTxtProgressBar(pb, i)
}

close(pb)


# --------------------------------

# 

# Train and save 10 NNs (20,10) using the 13 lag train_set

train_and_save_nn <- function(train_set, number_neurons, f, nn_name) {
  nn <- neuralnet(f, data = train_set, hidden = number_neurons, linear.output = TRUE)
  save(nn, file = paste0(nn_name, ".RData"))
}


num_models <- 10
number_neurons <- c(20, 10)
neurons_str <- paste(number_neurons, collapse = "_")


pb <- txtProgressBar(min = 1, max = num_models, style = 3)

for (i in 1:num_models) {
  nn_name <- paste0("nn_model_lag13__", neurons_str, "_", i)
  train_and_save_nn(train_set, number_neurons, f, nn_name)
  setTxtProgressBar(pb, i)
}

close(pb)

# train just one nn 20_10 to check the problems we are getting
set.seed(100)
nn_name <- paste0("nn_model_lag13__", neurons_str, "_check")
train_and_save_nn(train_set, number_neurons, f, nn_name)


#############################################################
estimate_nn<-function(train_set,number_neurons,data_mat,test_set,f)
{
  nn <- neuralnet(f,data=train_set,hidden=number_neurons,linear.output=T)
  
  
  # In sample performance
  predicted_scaled_in_sample<-nn$net.result[[1]]
  # Scale back from interval [0,1] to original log-returns
  predicted_nn_in_sample<-predicted_scaled_in_sample*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # In-sample MSE
  MSE.in.nn<-mean(((train_set[,1]-predicted_scaled_in_sample)*(max(data_mat[,1])-min(data_mat[,1])))^2)
  
  # Out-of-sample performance
  # Compute out-of-sample forecasts
  pr.nn <- predict(nn,as.matrix(test_set[,2:ncol(test_set)]))
  predicted_scaled<-pr.nn
  # Results from NN are normalized (scaled)
  # Descaling for comparison
  predicted_nn <- predicted_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  test.r <- test_set[,1]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
  MSE.out.nn <- mean((test.r - predicted_nn)^2)
  
  # Compare in-sample and out-of-sample
  MSE_nn<-c(MSE.in.nn,MSE.out.nn)
  return(list(MSE_nn=MSE_nn,predicted_nn=predicted_nn,predicted_nn_in_sample=predicted_nn_in_sample))
  
}

anzsim<-10
# set.seed(0)
number_neurons<-c(12,6)
MSE_mat<-matrix(ncol=2,nrow=anzsim)
colnames(MSE_mat)<-c("In sample MSE","Out sample MSE")

pb <- txtProgressBar(min = 1, max = anzsim, style = 3)

for (i in 1:anzsim)#i<-12
{
  MSE_mat[i,]<-estimate_nn(train_set,number_neurons,data_mat,test_set,f)$MSE_nn
  print(c(i,MSE_mat[i,]))
  setTxtProgressBar(pb, i)
  
}
close(pb)

head(train_set)
