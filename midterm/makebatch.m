function batchdata=makebatch(filename)
data=csvread(filename);
totnum=size(data,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rng(sum(100*clock)); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(data,2);
batchsize = 100;
batchdata1 = zeros(batchsize, numdims, numbatches);

for b=1:numbatches
  batchdata1(:,:,b) = data(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end
batchdata=batchdata1;
end
