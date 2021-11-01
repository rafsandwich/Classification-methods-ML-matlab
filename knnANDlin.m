close all;
clear all;

load fisheriris
X = meas;
Y = species;

%shuffling the data to approximately evenly distribute the classes
P = randperm(length(X));
X = X(P, :);
Y = Y(P); 

%initial visualisation of data as scatter plot on petal length & petal
%width
figure(1)
gscatter(X(:,3),X(:,4),Y,'rgb','osd');
xlabel('Petal length');
ylabel('Petal width');  

%P = X(:,3:4)

%hold on;
%scatter(5.4,0.2)

%data management
%4 fold quarter splits of first 1:120 data rows
%5th final test fold to be used for determining eventual performance

%training and validation
Xfold1 = X(1:30,1:4);
Xfold2 = X(31:60,1:4);
Xfold3 = X(61:90,1:4);
Xfold4 = X(91:120,1:4);

Yfold1 = Y(1:30,1);
Yfold2 = Y(31:60,1);
Yfold3 = Y(61:90,1);
Yfold4 = Y(91:120,1);

%final testing
Xfold5 = X(121:150,1:4);
Yfold5 = Y(121:150,1);
 
Xtrainset1 = [Xfold1;Xfold2;Xfold3]; %validate fold4
Xtrainset2 = [Xfold1;Xfold2;Xfold4]; %validate fold3
Xtrainset3 = [Xfold1;Xfold3;Xfold4]; %validate fold2
Xtrainset4 = [Xfold2;Xfold3;Xfold4]; %validate fold1

Ytrainset1 = [Yfold1;Yfold2;Yfold3]; %validate fold4
Ytrainset2 = [Yfold1;Yfold2;Yfold4]; %validate fold3
Ytrainset3 = [Yfold1;Yfold3;Yfold4]; %validate fold2
Ytrainset4 = [Yfold2;Yfold3;Yfold4]; %validate fold1

k = findBestK(X,Y) %value of k, number of neighbours 

%begin training, fitcknn, TRAINING SET 1

knn = fitcknn(Xtrainset1,Ytrainset1,'NumNeighbors',k,'Standardize',1);
Ytrainset1_predict = predict(knn,Xtrainset1); %training error
Yfold4_predict = predict(knn,Xfold4); %validation

%training, set 1, error rate
train1_ER = errorcalc(Ytrainset1, Ytrainset1_predict);
%validation, set 1, error rate
test1_ER = errorcalc(Yfold4, Yfold4_predict);

% TRAINING SET 2

knn = fitcknn(Xtrainset2,Ytrainset2,'NumNeighbors',k,'Standardize',1);
Ytrainset2_predict = predict(knn,Xtrainset2);
Yfold3_predict = predict(knn,Xfold3);

train2_ER = errorcalc(Ytrainset2, Ytrainset2_predict);
test2_ER = errorcalc(Yfold3, Yfold3_predict);

% TRAINING SET 3

knn = fitcknn(Xtrainset3,Ytrainset3,'NumNeighbors',k,'Standardize',1);
Ytrainset3_predict = predict(knn,Xtrainset3);
Yfold2_predict = predict(knn,Xfold2); 

train3_ER = errorcalc(Ytrainset3, Ytrainset3_predict);
test3_ER = errorcalc(Yfold2, Yfold2_predict);

% TRAINING SET 4
 
knn = fitcknn(Xtrainset4,Ytrainset4,'NumNeighbors',k,'Standardize',1);
Ytrainset4_predict = predict(knn,Xtrainset4);
Yfold1_predict = predict(knn,Xfold1);

train4_ER = errorcalc(Ytrainset4, Ytrainset4_predict);
test4_ER = errorcalc(Yfold1, Yfold1_predict);

% knn cross validation error calculation
KNNcrossvalidation_ER = (test1_ER + test2_ER + test3_ER + test4_ER)/4


%begin training, fitcecoc, TRAINING SET 1

LINclass = fitcecoc(Xtrainset1,Ytrainset1);
Ytrainset1_predictLIN = predict(LINclass,Xtrainset1); %training error
Yfold4_predictLIN = predict(LINclass,Xfold4); %validation

%training, set 1, error rate
LINtrain1_ER = errorcalc(Ytrainset1, Ytrainset1_predictLIN);
%validation, set 1, error rate
LINtest1_ER = errorcalc(Yfold4, Yfold4_predictLIN);

% TRAINING SET 2

LINclass = fitcecoc(Xtrainset2,Ytrainset2);
Ytrainset2_predictLIN = predict(LINclass,Xtrainset2);
Yfold3_predictLIN = predict(LINclass,Xfold3);

LINtrain2_ER = errorcalc(Ytrainset2, Ytrainset2_predictLIN);
LINtest2_ER = errorcalc(Yfold3, Yfold3_predictLIN);

% TRAINING SET 3

LINclass = fitcecoc(Xtrainset3,Ytrainset3);
Ytrainset3_predictLIN = predict(LINclass,Xtrainset3);
Yfold2_predictLIN = predict(LINclass,Xfold2); 

LINtrain3_ER = errorcalc(Ytrainset3, Ytrainset3_predictLIN);
LINtest3_ER = errorcalc(Yfold2, Yfold2_predictLIN);

% TRAINING SET 4
 
LINclass = fitcecoc(Xtrainset4,Ytrainset4);
Ytrainset4_predictLIN = predict(LINclass,Xtrainset4);
Yfold1_predictLIN = predict(LINclass,Xfold1);

LINtrain4_ER = errorcalc(Ytrainset4, Ytrainset4_predictLIN);
LINtest4_ER = errorcalc(Yfold1, Yfold1_predictLIN);

% cross validation error calculation
LINcrossvalidation_ER = (LINtest1_ER + LINtest2_ER + LINtest3_ER + LINtest4_ER)/4

%deciding linear classification is better suited as it performed best at
%cross validation
%creating a new model that uses more of the data in a total set (1:120)
% and finally tests against untouched fold 5 (120:150)

Xfinaltrain = [Xfold1;Xfold2;Xfold3;Xfold4];
Yfinaltrain = [Yfold1;Yfold2;Yfold3;Yfold4];

LINclass2 = fitcecoc(Xfinaltrain,Yfinaltrain);
Yfinaltrainset_predict = predict(LINclass2,Xfinaltrain); %training error
Yfold5_predict = predict(LINclass,Xfold5); %validation

%final test predicting error rate between unused fold 5 and predicted fold 5
%labels
finaltest_ER = errorcalc(Yfold5, Yfold5_predict)

finalYpred = [Yfinaltrainset_predict;Yfold5_predict];
%this second figure shows how the classifier model predicted the species
%and can be looked at next to figure(1) to visually see the slight
%differences when there is some prediction error
figure(2)
gscatter(X(:,3),X(:,4),finalYpred,'rgb','osd');

%calculates the error rate between a predicted set and its true labels
function[errorRate] = errorcalc(Yset, Ypred)
    foldlength = length(Yset);
    %number of errors occured
    e = 0;
    for i = 1:foldlength
        if isequal(Ypred(i,1), Yset(i,1))
            %do nothing, correct label assigned
        else
            %else label is wrong, count an error
            e = e+1;
        end
    end
    %calculate mean error rate, number of errors divided by number of entries
    errorRate = e/foldlength;
end

%function to find the value of K that results in the lowest cross val error
%rate
function[bestK] = findBestK(X,Y)
    cv_ER = 100;
    
    Xfold1 = X(1:30,1:4);
    Xfold2 = X(31:60,1:4);
    Xfold3 = X(61:90,1:4);
    Xfold4 = X(91:120,1:4);
    
    Yfold1 = Y(1:30,1);
    Yfold2 = Y(31:60,1);
    Yfold3 = Y(61:90,1);
    Yfold4 = Y(91:120,1);
    
    Xtrainset1 = [Xfold1;Xfold2;Xfold3]; %validate fold4
    Xtrainset2 = [Xfold1;Xfold2;Xfold4]; %validate fold3
    Xtrainset3 = [Xfold1;Xfold3;Xfold4]; %validate fold2
    Xtrainset4 = [Xfold2;Xfold3;Xfold4]; %validate fold1
    
    Ytrainset1 = [Yfold1;Yfold2;Yfold3]; %validate fold4
    Ytrainset2 = [Yfold1;Yfold2;Yfold4]; %validate fold3
    Ytrainset3 = [Yfold1;Yfold3;Yfold4]; %validate fold2
    Ytrainset4 = [Yfold2;Yfold3;Yfold4]; %validate fold1
    for i = 1:90
        knn = fitcknn(Xtrainset1,Ytrainset1,'NumNeighbors',i,'Standardize',1);
        Yfold4_predict = predict(knn,Xfold4); 
        
        test1_ER = errorcalc(Yfold4, Yfold4_predict);
        
        % TRAINING SET 2
        
        knn = fitcknn(Xtrainset2,Ytrainset2,'NumNeighbors',i,'Standardize',1);
        Yfold3_predict = predict(knn,Xfold3);
      
        test2_ER = errorcalc(Yfold3, Yfold3_predict);
        
        % TRAINING SET 3
        
        knn = fitcknn(Xtrainset3,Ytrainset3,'NumNeighbors',i,'Standardize',1);
        Yfold2_predict = predict(knn,Xfold2);
        
        test3_ER = errorcalc(Yfold2, Yfold2_predict);
        
        % TRAINING SET 4
        
        knn = fitcknn(Xtrainset4,Ytrainset4,'NumNeighbors',i,'Standardize',1);
        Yfold1_predict = predict(knn,Xfold1);

        test4_ER = errorcalc(Yfold1, Yfold1_predict);
        
        % cross validation error calculation
        crossvalidation_ER = (test1_ER + test2_ER + test3_ER + test4_ER)/4;
        if crossvalidation_ER < cv_ER %if this ER is smaller than previous smallest
            cv_ER = crossvalidation_ER; %%this ER becomes new smallest
            bestK=i; %best K is updated with current value
        end
    end
    fprintf('Best K classifcation error rate = %f\n', cv_ER); %prints best error rate to command window
end
        
