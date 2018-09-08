#import data
library(readxl)
CTGData <- read_excel("F:/Files/DataMining/HW1/CTGData.xlsx",
     sheet = "Data", col_types = c("blank", 
                                    "blank", "blank", "blank", "blank", 
                                    "blank", "blank", "blank", "blank", 
                                    "blank", "numeric", "numeric", "numeric", 
                                    "numeric", "numeric", "numeric", 
                                    "numeric", "numeric", "numeric", 
                                    "numeric", "numeric", "blank", "blank", 
                                    "blank", "blank", "blank", "blank", 
                                    "blank", "blank", "blank", "blank", "blank", 
                                    "blank", "blank", "blank", "blank", "blank", 
                                    "blank", "blank", "blank", "blank", "blank", 
                                    "blank", "text", "blank", "blank"))
View(CTGData)

CTGData=CTGData[-1,1:12]
class1=CTGData[which(CTGData[12]==1|CTGData[12]==2),1:12]
class2=CTGData[which(CTGData[12]==3|CTGData[12]==4),1:12]
class3=CTGData[which(CTGData[12]==5|CTGData[12]==6),1:12]
class4=CTGData[which(CTGData[12]==7|CTGData[12]==8),1:12]
class5=CTGData[which(CTGData[12]==9|CTGData[12]==10),1:12]
hw1=rbind(class1,class2,class3,class4,class5)

#numerical characterics
chard=array(0,c(11,5))
for (i in 1:11){
  chard[i,]=c(max(hw1[i]),min(hw1[i]),max(hw1[i])-min(hw1[i]),mean(as.matrix(hw1[i])),sqrt(var(hw1[i])))
}


#histograms
for (i in 1:11)
{
  setwd("F:\\Files\\DataMining\\HW1")
  jpeg(file=paste("D",i,".jpeg"))
  hist(as.matrix(hw1[i]))
  dev.off()
}
par(mfrow=c(2,6))
for (i in 1:11){
  hist(as.matrix(hw1[i]),main=paste("Hist of descriptor",i))
}

par(mfrow=c(2,3))
for (i in 1:6){
hist(as.matrix(class1[i]),freq=FALSE,main=paste("descriptor",i," & Class"),xlab="",xlim=c(min(hw1[i]),max(hw1[i])),ylim=c(0,10/chard[i,3]),breaks=seq(min(hw1[i]),max(hw1[i]),by=chard[i,3]/9),col=rgb(230,25,75, maxColorValue=255,alpha=100))
hist(as.matrix(class2[i]),freq=FALSE,add=T,col=rgb(60,180,75, maxColorValue=255,alpha=100),breaks=seq(min(hw1[i]),max(hw1[i]),by=(chard[i,3])/9))
hist(as.matrix(class3[i]),freq=FALSE,add=T,col=rgb(255,255,25, maxColorValue=255,alpha=100),breaks=seq(min(hw1[i]),max(hw1[i]),by=(chard[i,3])/9))
hist(as.matrix(class4[i]),freq=FALSE,add=T,col=rgb(0,130,200, maxColorValue=255,alpha=100),breaks=seq(min(hw1[i]),max(hw1[i]),by=(chard[i,3])/9))
hist(as.matrix(class5[i]),freq=FALSE,add=T,col=rgb(145,30,180, maxColorValue=255,alpha=100),breaks=seq(min(hw1[i]),max(hw1[i]),by=(chard[i,3])/9))
}

par(mfrow=c(2,3))
for (i in 7:11){
  hist(as.matrix(class1[i]),freq=FALSE,main=paste("descriptor",i," & Class"),xlab="",xlim=c(min(hw1[i]),max(hw1[i])),ylim=c(0,10/chard[i,3]),breaks=seq(min(hw1[i]),max(hw1[i]),by=chard[i,3]/9),col=rgb(230,25,75, maxColorValue=255,alpha=100))
  hist(as.matrix(class2[i]),freq=FALSE,add=T,col=rgb(60,180,75, maxColorValue=255,alpha=100),breaks=seq(min(hw1[i]),max(hw1[i]),by=(chard[i,3])/9))
  hist(as.matrix(class3[i]),freq=FALSE,add=T,col=rgb(255,255,25, maxColorValue=255,alpha=100),breaks=seq(min(hw1[i]),max(hw1[i]),by=(chard[i,3])/9))
  hist(as.matrix(class4[i]),freq=FALSE,add=T,col=rgb(0,130,200, maxColorValue=255,alpha=100),breaks=seq(min(hw1[i]),max(hw1[i]),by=(chard[i,3])/9))
  hist(as.matrix(class5[i]),freq=FALSE,add=T,col=rgb(145,30,180, maxColorValue=255,alpha=100),breaks=seq(min(hw1[i]),max(hw1[i]),by=(chard[i,3])/9))
}

#ks & t test
class=list(class1,class2,class3,class4,class5)
t=array(0,c(5,5,11))
ks=array(0,c(5,5,11))
mc=array(0,c(11,5))
sdc=array(0,c(11,5))
for (i in 1:11){
  for (k in 1:5){
    mc[i,k]=mean(class[[k]][[i]])
    sdc[i,k]=sqrt(var(class[[k]][[i]]))
    for (m in 1:5){
      t[k,m,i]=t.test(as.matrix(class[[k]][[i]]),as.matrix(class[[m]][[i]]))$p.value
      ks[k,m,i]=ks.test(as.matrix(class[[k]][[i]]),as.matrix(class[[m]][[i]]))$p.value
      }
  }
}


#pca
pca1=prcomp(as.matrix(hw1[1:11]),tol=sqrt(.Machine$double.eps))
lambda=pca1$sdev*pca1$sdev
ratio=array(0,11)
for (j in 1:11){
  ratio[j]=sum(lambda[1:j])/sum(lambda)}
#  if (ratio[j]>=0.95)
#    break
w1=as.matrix(pca1$rotation[,1])
w2=as.matrix(pca1$rotation[,2])
w3=as.matrix(pca1$rotation[,3])
temp=as.matrix(hw1[1:11])
projection1=array(0,c(3,963))
projection2=array(0,c(3,134))
projection3=array(0,c(3,404))
projection4=array(0,c(3,359))
projection5=array(0,c(3,266))
for (i in 1:963){
  projection1[1,i]=as.numeric(temp[i,])%*%w1
  projection1[2,i]=as.numeric(temp[i,])%*%w2
  projection1[3,i]=as.numeric(temp[i,])%*%w3
}
for (i in 1:134){
  projection2[1,i]=as.numeric(temp[(i+963),])%*%w1
  projection2[2,i]=as.numeric(temp[(i+963),])%*%w2
  projection2[3,i]=as.numeric(temp[(i+963),])%*%w3
}
for (i in 1:404){
  projection3[1,i]=as.numeric(temp[(i+1097),])%*%w1
  projection3[2,i]=as.numeric(temp[(i+1097),])%*%w2
  projection3[3,i]=as.numeric(temp[(i+1097),])%*%w3
}
for (i in 1:359){
  projection4[1,i]=as.numeric(temp[(i+1501),])%*%w1
  projection4[2,i]=as.numeric(temp[(i+1501),])%*%w2
  projection4[3,i]=as.numeric(temp[(i+1501),])%*%w3
}
for (i in 1:266){
  projection5[1,i]=as.numeric(temp[(i+1760),])%*%w1
  projection5[2,i]=as.numeric(temp[(i+1760),])%*%w2
  projection5[3,i]=as.numeric(temp[(i+1760),])%*%w3
}
plot3d(projection1[1,],projection1[2,],projection1[3,],col=rgb(230,25,75, maxColorValue=255))
plot3d(projection2[1,],projection2[2,],projection2[3,],add=T,col=rgb(60,180,75, maxColorValue=255))
plot3d(projection3[1,],projection3[2,],projection3[3,],add=T,col=rgb(255,255,25, maxColorValue=255))
plot3d(projection4[1,],projection4[2,],projection4[3,],add=T,col=rgb(0,130,200, maxColorValue=255))
plot3d(projection5[1,],projection5[2,],projection5[3,],add=T,col=rgb(145,30,180, maxColorValue=255))

plot(projection1[1,],projection1[2,],col=rgb(230,25,75, maxColorValue=255),'p')
points(projection2[1,],projection2[2,],col=rgb(60,180,75, maxColorValue=255))
points(projection3[1,],projection3[2,],col=rgb(255,255,25, maxColorValue=255))
points(projection4[1,],projection4[2,],col=rgb(0,130,200, maxColorValue=255))
points(projection5[1,],projection5[2,],col=rgb(145,30,180, maxColorValue=255))

plot(projection1[1,],projection1[3,],col=rgb(230,25,75, maxColorValue=255),'p')
points(projection2[1,],projection2[3,],col=rgb(60,180,75, maxColorValue=255))
points(projection3[1,],projection3[3,],col=rgb(255,255,25, maxColorValue=255))
points(projection4[1,],projection4[3,],col=rgb(0,130,200, maxColorValue=255))
points(projection5[1,],projection5[3,],col=rgb(145,30,180, maxColorValue=255))

plot(projection1[3,],projection1[2,],col=rgb(230,25,75, maxColorValue=255),'p')
points(projection2[3,],projection2[2,],col=rgb(60,180,75, maxColorValue=255))
points(projection3[3,],projection3[2,],col=rgb(255,255,25, maxColorValue=255))
points(projection4[3,],projection4[2,],col=rgb(0,130,200, maxColorValue=255))
points(projection5[3,],projection5[2,],col=rgb(145,30,180, maxColorValue=255))
