%Clear screen

clc
clear all
close all

%Load data

load veri

%An example data

%figure;
%plot(veri(222,:))

%Define attributes

attribute = zeros(5,length(etiket));

for j=1:length(etiket)
    
    x = veri(j,:);
    attribute(1,j) = (1/length(x))*sum(x.^2);
    attribute(2,j) = std(x);
    attribute(3,j) = mean(abs(x));
    attribute(4,j) = skewness(x);
    attribute(5,j) = kurtosis(x);
    
end

%Cross Validation

fold = cvpartition(etiket, "kfold", 4);

%Split training and test data

label = etiket;

for i = 1:4

    trainIdx = fold.training(i);
    testIdx = fold.test(i);

    xTrain = attribute(:,trainIdx);
    yTrain = label(trainIdx);

    xTest = attribute(:,testIdx);
    yTest = label(testIdx);

    %Prepare template for SVM

    t = templateSVM('Standardize', true, 'KernelFunction', 'polynomial');

    %Ready input

    input = transpose(xTrain);

    %Fit SVM model

    SVMModel = fitcecoc(input,yTrain,'Learners',t,'FitPosterior',true ,'ClassNames',{'Cars','Drones','People'});

    %Predict yTest values

    predictions = predict(SVMModel,transpose(xTest));
    predictions = categorical(cellstr(predictions));

    %Digitize predictions and actual data for model metric calculation

    digitizedyTrain = zeros(length(yTest),1);
    digitizedPredicted = zeros(length(yTest),1);

    for j=1:length(yTest)
       if yTest(j,1) == "Cars"
           digitizedyTrain(j) = 1;
       end
       if yTest(j,1) == "Drones"
           digitizedyTrain(j) = 2;
       end
       if yTest(j,1) == "People"
           digitizedyTrain(j) = 3;
       end
       if predictions(j,1) == "Cars"
           digitizedPredicted(j) = 1;
       end
       if predictions(j,1) == "Drones"
           digitizedPredicted(j) = 2;
       end
       if predictions(j,1) == "People"
           digitizedPredicted(j) = 3;
       end
    end

    %Draw confusion chart

    cm = confusionchart(yTest,predictions,'RowSummary','row-normalized','ColumnSummary','column-normalized','Title','Eş Oluşum Matrisi');

    %Print model metrics

    [c_matrix, Result, ReferanceResult] = confusion.getMatrix(digitizedyTrain, digitizedPredicted);
    
    sensitivity(1, i) = Result.Sensitivity*100;
    specificity(1, i) = Result.Specificity*100;
    acc(1, i) = Result.Accuracy*100;
    precision(1, i) = Result.Precision*100;
    F1_score(1,i) = Result.F1_score*100;
    MCC(1,i) = Result.MatthewsCorrelationCoefficient*100;
    Kappa(1, i) = Result.Kappa*100;
    
    result{1, i} = Result;
    SVM{1, i} = SVMModel;

end


mean_sensitivity_svm = mean(sensitivity)
mean_specificity_svm = mean(specificity)
mean_accuracy_svm = mean(acc)
mean_precision_svm = mean(precision)
mean_F1_score_svm = mean(F1_score)
mean_MCC_svm = mean(MCC)

for i = 1:4

    trainIdx = fold.training(i);
    testIdx = fold.test(i);

    xTrain = attribute(:,trainIdx);
    yTrain = label(trainIdx);

    xTest = attribute(:,testIdx);
    yTest = label(testIdx);
    
    for j=1:length(yTrain)
       if yTrain(j,1) == "Cars"
           digitizedyTrain(j) = 1;
       end
       if yTrain(j,1) == "Drones"
           digitizedyTrain(j) = 2;
       end
       if yTrain(j,1) == "People"
           digitizedyTrain(j) = 3;
       end
    end
    
    for j=1:length(yTest)
       if yTest(j,1) == "Cars"
           digitizedyTest(j) = 1;
       end
       if yTest(j,1) == "Drones"
           digitizedyTest(j) = 2;
       end
       if yTest(j,1) == "People"
           digitizedyTest(j) = 3;
       end
    end

    %Prepare template for ANN
    
    net = feedforwardnet(20);

    net.trainParam.epochs = 500;
    net.trainParam.lr = 0.9;
    net.trainParam.mc = 0.9;
    net.trainParam.max_fail = 500;
    
    %Train the ANN
    
    [net, tr] = train(net, xTrain, digitizedyTrain);
    
    %Test the ANN
    
    [outputTest] = sim(net, xTest);
    
    %Round the predictions
    
    for k=1:length(outputTest)
       
        if outputTest(1,k) < 1.5;
            roundedoutputTest(1,k) = 1;
        end
        if outputTest(1,k) > 1.5 && outputTest(1,k) < 2.5;
            roundedoutputTest(1,k) = 2;
        end
        if outputTest(1,k) > 2.5 && outputTest(1,k) < 3.5;
            roundedoutputTest(1,k) = 3;
        end
    end
    
    %Print model metrics
    
    [c_matrix, Result, ReferanceResult] = confusion.getMatrix(digitizedyTest, roundedoutputTest);
   
    sensitivity(1, i) = Result.Sensitivity*100;
    specificity(1, i) = Result.Specificity*100;
    acc(1, i) = Result.Accuracy*100;
    precision(1, i) = Result.Precision*100;
    F1_score(1,i) = Result.F1_score*100;
    MCC(1,i) = Result.MatthewsCorrelationCoefficient*100;
    Kappa(1, i) = Result.Kappa*100;
    
    result{1, i} = Result;
    ann{1, i} = net;
    
end

%Get the average of metrics

mean_sensitivity_ann = mean(sensitivity)
mean_specificity_ann = mean(specificity)
mean_accuracy_ann = mean(acc)
mean_precision_ann = mean(precision)
mean_F1_score_ann = mean(F1_score)
mean_MCC_ann = mean(MCC)
