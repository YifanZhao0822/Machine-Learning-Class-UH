% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 
clear;clc;
restart=1;
epsilonw      = 0.1;   % Learning rate for weights 
epsilonvb     = 0.1;   % Learning rate for biases of visible units 
epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
weightcost  = 0;   
initialmomentum  = 0;
finalmomentum    = 0;
batchdata=makebatch('bidata.csv');
numhid=91;
maxepoch=30;

totdata=csvread('bidata.csv');
totnum=size(totdata,1);
numbatches=totnum/100;
numdims  =  size(totdata,2);
batchsize = 100;
%[batchsize,numdims,numbatches]=size(batchdata);


if restart ==1
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases.
  %inhid=0.1*randn(numdims,numhid);
  vishid     = 0.1*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);

  poshidprobs = zeros(batchsize,numhid);
  neghidprobs = zeros(batchsize,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  %inhidinc  = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(batchsize,numhid,numbatches);
  global_step=1;
  gradient=zeros(maxepoch);
  meanwei=zeros(maxepoch);
  stdevwei=zeros(maxepoch);
end

for epoch = 1:maxepoch
 batchdata=zeros(batchsize, numdims, numbatches);
 rng(sum(100*clock));
 randomorder=randperm(totnum);
 for b=1:numbatches
  batchdata(:,:,b) = totdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
 end

 %batchdata=makebatch('bidata.csv');
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches
 fprintf(1,'epoch %d batch %d\r',epoch,batch); 

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,batchsize,1)));    
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;
  %posinprods=data'*poshidprobs;
  poshidact   = sum(poshidprobs);
  posvisact = sum(data);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(batchsize,numhid);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,batchsize,1)));
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,batchsize,1)));    
  negprods  = negdata'*neghidprobs;
  %neginprods=data'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 ));
  errsum = err + errsum;

   if epoch>5
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    %inhidinc=epsilonw*( (posinprods-neginprods)/numcases);
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/batchsize - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/batchsize)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/batchsize)*(poshidact-neghidact);
    
    %inhid=inhid+inhidinc;
    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  gradient(epoch)=norm(vishidinc/epsilonw);
  meanwei(epoch)=mean(mean(vishid));
  stdevwei(epoch)=std(reshape(vishid,[1,numdims*numhid]));
  global_step=global_step+1;
 end
 
 if rem(epoch,10)==0
  figure
  histogram(vishid);
  title(['Histogram of Weights after ' epoch ' epoch']);
 end
 
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, err); 
  if gradient(epoch)<2.1
      break
  end
end
figure
plot(1:maxepoch, gradient)
title('Gradient')
figure
plot(1:maxepoch, meanwei)
title('Means of Weights')
figure
plot(1:maxepoch, stdevwei)
title('Standard Deviations of Weights')
figure
histogram(vishid);
title('Histogram of Final Weights');

data=csvread('bidata.csv');
numcases=size(data,1);
out=zeros(numcases,numdims,21);
aveout=zeros(numcases);
for n=1:numcases
    hidden=1./(1 + exp(-data(n,:)*vishid - hidbiases));
    out(n,:,1)=1./(1 + exp(-hidden*vishid' - visbiases));
    for i=1:20
        hidden=1./(1 + exp(-out(n,:,i)*vishid - hidbiases));
        out(n,:,i+1)=1./(1 + exp(-hidden*vishid' - visbiases));
    end
    aveout(n)=norm(mean(out(n,:,:),3)-data(n,:))^2/numdims;
end
aveout=aveout(:,1);
MSE=mean(aveout);