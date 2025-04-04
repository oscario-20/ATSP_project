# Installiere und lade benötigte Pakete
if (!require(quantmod)) install.packages("quantmod", dependencies = TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
# Lade die Pakete
library(quantmod)
library(ggplot2)
library(dplyr)

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




