clc;
close all;
clear all;

% this is unsupervised classification using clustering algorithm. each
% class of data is modelled into separate cluster model.
%% Training Side
% three class data -- each [5X3] 5 observation with 3 features
dataclass1 = [1 2 33;
              4 3 42;
              5 7 33;
              4 5 34;
              2 3 35]';
dataclass2 = [2 44 3;
              3 55 1;
              2 47 2;
              4 44 5;
              2 55 3]';
dataclass3 = [44 4 5;
              55 2 3;
              49 4 4;
              53 5 3;
              59 4 2]';

%create a hidden markov model
  O = 3;      %Number of features
  T = 5;      %Number of observations 
  nex = 1;    %Number of sequences 
  M = 1;      %Number of mixtures 
  Q = 2;      %Number of states 
  cov_type = 'full';

% Create class model for each class of data
[prior0, transmat0, mu0, Sigma0, mixmat0] = hmminitialize(dataclass1,O,T,nex,M,Q,cov_type);
[LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    mhmm_em(dataclass1, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 5);% Fitting category1 to object 1
[prior0, transmat0, mu0, Sigma0, mixmat0] = hmminitialize(dataclass2,O,T,nex,M,Q,cov_type);
[LL, prior2, transmat2, mu2, Sigma2, mixmat2] = ...
    mhmm_em(dataclass2, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 5);% Fitting category1 to object 2
[prior0, transmat0, mu0, Sigma0, mixmat0] = hmminitialize(dataclass3,O,T,nex,M,Q,cov_type);
[LL, prior3, transmat3, mu3, Sigma3, mixmat3] = ...
    mhmm_em(dataclass3, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 5);% Fitting category1 to object 3

obj1 = [{prior1}, {transmat1}, {mu1}, {Sigma1}, {mixmat1}];
obj2 = [{prior2}, {transmat2}, {mu2}, {Sigma2}, {mixmat2}];
obj3 = [{prior3}, {transmat3}, {mu3}, {Sigma3}, {mixmat3}];
% obj is the trained model. save it at end for doing testing
save('obj1.mat','obj1');
save('obj2.mat','obj2');
save('obj3.mat','obj3');

% train performance
% likelihood of a particular data observations for each model is
% calculated and the observation is labelled to the model with maximum
% likelihood
trainperfdata=[dataclass1';dataclass2';dataclass3']';
for i=1:size(trainperfdata,2)
a(i) = mhmm_logprob(trainperfdata(:,i), obj1{1}, obj1{2}, obj1{3}, obj1{4}, obj1{5});
b(i) = mhmm_logprob(trainperfdata(:,i), obj2{1}, obj2{2}, obj2{3}, obj2{4}, obj2{5});
c(i) = mhmm_logprob(trainperfdata(:,i), obj3{1}, obj3{2}, obj3{3}, obj3{4}, obj3{5});
end
y =[a' b' c'];

[~,ymax] = max(y,[],2);
% original output expected for each data observation
target=[1;1;1;1;1;2;2;2;2;2;3;3;3;3;3];
performance = sum(target==ymax)/size(target,1) % performance in the range of 0 to 1

%% Testing Side
% for testing load the trained model
load('obj1.mat');
load('obj2.mat');
load('obj3.mat');
testdata = [4 5 34]'; % take 1 new unknown observation and give to trained model
a = mhmm_logprob(testdata, obj1{1}, obj1{2}, obj1{3}, obj1{4}, obj1{5});
b = mhmm_logprob(testdata, obj2{1}, obj2{2}, obj2{3}, obj2{4}, obj2{5});
c = mhmm_logprob(testdata, obj3{1}, obj3{2}, obj3{3}, obj3{4}, obj3{5});
o =[a' b' c'];

[~,omax] = max(o,[],2)
