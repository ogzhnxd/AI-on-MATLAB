%% Cleaning console and window
clc
clear all
close all

%% Importing data

dataPath = "waterQuality1.csv";
data = readtable(dataPath);

%% Cleaning data

% for i = 1:10
%     cleanData(:,i) = fillmissing(data(:,i),'movmean',10);
% end
%% Digitizing data

data = table2array(data);

%% Spliting inputs and outputs

inputs = transpose(data(:,1:20));
outputs = transpose(data(:,21));

%% Cross Validation

fold = cvpartition(outputs, 'kfold', 4);

%% Seperating test and train sets

for i = 1:4
    trainIdx = fold.training(i);
    testIdx = fold.test(i);
    
    xTrain = inputs(:,trainIdx);
    yTrain = outputs(:,trainIdx);
    
    xTest = inputs(:,testIdx);
    yTest = outputs(:,testIdx);

%% Forming the ANN

    net = feedforwardnet(5);

    net.trainParam.epochs = 100;
    net.trainParam.lr = 0.9;
    net.trainParam.mc = 0.9;
    net.trainParam.max_fail = 100;

%% Training the ANN

    [net, tr] = train(net, xTrain, yTrain);

%% Testing the ANN

    [outputTest] = sim(net, xTest);
    outputTest = round(outputTest);
    outputTest(outputTest>=1) = 1;
    outputTest(outputTest<=0) = 0;

%% Calculating metrics

    [c_matrix, Result, ReferanceResult] = confusion.getMatrix(yTest, outputTest);
   
    sensitivity(1, i) = Result.Sensitivity*100;
    specificity(1, i) = Result.Specificity*100;
    acc(1, i) = Result.Accuracy*100;
    precision(1, i) = Result.Precision*100;
    F_score(1,i) = Result.MatthewsCorrelationCoefficient*100;
    Kappa(1, i) = Result.Kappa*100;
    
    result{1, i} = Result;
    ann{1, i} = net;
    
end
