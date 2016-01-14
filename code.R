
# Load data
setwd('/Users/doutre/Desktop/Homework3')
data.test=read.table('./hw3-2-test.data')
names(data.test)=c('v1','v2','v3','y')
data.train=read.table('./hw3-2-train.data')
names(data.train)=c('v1','v2','v3','y')
sum(data.train$y==1)

# Set global variables
k=max(data.train$y)
d=3
n=dim(data.train)[1]

############## Logistic Regression #############

# Init N matrix
init_N=function(){
  N=matrix(0,ncol=k,nrow=n)
  for (i in 1:n){
    N[i,data.train$y[i]]=1
  }
  N
}

Pi = function(i,j,beta,x){
  xi=x[i,]
  num=exp(-t(beta[(1+(j-1)*d):(j*d),])%*%xi)
  den=1+sum(sapply(1:(k-1),function(i) exp(-t(beta[(1+(i-1)*d):(i*d),])%*%xi)))
  return(num/den)
}

gradient = function(beta,data){
  res=matrix(0,ncol=1,nrow=d*(k-1))
  x=as.matrix(data[,1:3])
  for (j in 1:(k-1)){
    res_i=matrix(0,ncol=1,nrow=d)
    for (i in 1:n){
      ni=sum(N[i,])
      nij=N[i,j]
      pi_ij=Pi(i,j,beta,x)
      res_i=res_i+x[i,]*(ni*pi_ij-nij)
    }
    res[(1+(j-1)*d):(j*d),]=res_i
  }
  return(res)
}

hessien = function(beta,data){
  x=as.matrix(data[,1:3])
  res=matrix(0,ncol=d*(k-1),nrow=d*(k-1))
  for (i in 1:n){
    ni=sum(N[i,])
    Pi=sapply(1:(k-1),function(j) {Pi(i,j,beta,x)})
    Pi_matrix=as.matrix(Pi,ncol=1)
    Di=ni*(Pi_matrix%*%t(Pi_matrix)-diag(Pi))
    Xi=matrix(0,ncol=k-1,nrow=d*(k-1))
    xi=x[i,]
    for (j in 1:(k-1) ){
      Xi[(1+(j-1)*d):(j*d),j]=xi
    }
    res=res+Xi%*%Di%*%t(Xi)
  }
  return(res)
}

newton_raphson = function(beta0,eps,rho,data){
  beta=beta0
  s=matrix(eps+1,1)
  i=1
  print('Reaching optimum...')
  while(norm(s,"1")>eps){
    H=hessien(beta,data.train)
    G=gradient(beta,data.train)
    s=solve(H,G)
    beta=beta-rho*s
    i=i+1
  }
  list(beta=beta,diff=s,iteration=i)
}
f1=function(x,y) {
  m=matrix(cbind(x,y,rep(1,length(x))),ncol=d)
  p=as.matrix(
    sapply(1:length(x),function(i) 
      sapply(1:(k-1),function(j) Pi(i,j,beta,m))
    )
  )
  s=colSums(p)
  m=rbind(p,1-s)
  apply(m,2,which.max)
}
error_rate2=function(beta,data){
  1-mean(f1(data.train[,1],data.train[,2])==data$y)
}
# Algorithm
N=init_N()
beta0=matrix(1,nrow=d*(k-1),ncol=1)
nr=newton_raphson(beta0,0.1,0.1)
beta=nr$beta

# Training Error
error_rate(beta,data.train)
# Test Error
error_rate(beta,data.test)
# Plotting function

# Plot
x=seq(-7,2,0.02)
contour(x,x,outer(x,x,f1),levels=c(1,2,3,4))
points(data.train[,1:2],col=data.test$y)
title(paste('Logistic Regression\nMissclassification test error rate :',
            round(error_rate(beta,data.test)*100),"%"))
# Table
table(f1(data.train[,1],data.train[,2]),data.test$y)


############## Linear Regression #############
linear_estimate=function(data){
  x=as.matrix(data[,1:3])
  names(x)=NULL
  y0=as.matrix(data$y)
  y=matrix(0,nrow=n,ncol=k)
  for (i in 1:n){
    y[i,y0[i,]]=1
  }
  X=matrix(0,d,d)
  for (i in 1:n){
    X=X+x[i,]%*%t(x[i,])
  }
  XY=matrix(0,d,k)
  for (i in 1:n){
    xi=as.matrix(x[i,])
    yi=as.matrix(y[i,])
    XY=XY+xi%*%t(yi)
  }
  Beta=2*solve(X,XY)
  row.names(Beta)=NULL
  return(Beta)
}


f2=function(x,y) {
  m=matrix(cbind(x,y,rep(1,length(x))),ncol=d)%*%beta
  res=matrix(0,ncol=1,nrow=length(x))
  for (i in 1:length(x)){
    res[i,]=which.max(m[i,])
  }
  res
}
error_rate2=function(beta,data){
  1-mean(f2(data.train[,1],data.train[,2])==data$y)
}

# Estimate
beta=linear_estimate(data.train)
# Training Error
error_rate2(beta,data.train)
# Test Error
error_rate2(beta,data.test)
# Plot : test
x=seq(-7,2,0.01)
contour(x,x,outer(x,x,f2),levels=c(1,2,3,4))
points(data.train[,1:2],col=data.test$y)
title(paste('Linear Regression\nMissclassification test error rate :',
            round(error_rate2(beta,data.test)*100),"%"))
# Table
table(f2(data.train[,1],data.train[,2]),data.test$y)
       