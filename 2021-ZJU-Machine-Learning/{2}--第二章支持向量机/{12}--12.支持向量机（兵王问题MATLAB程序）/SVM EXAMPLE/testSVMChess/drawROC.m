%Test the model on the remaining testing data and obtain the recognition rate.
clear all;
load model.mat;
load xTesting.mat;
load yTesting.mat;
[yPred,accuracy,decisionValues] = svmpredict(yTesting,xTesting,model); 

%draw ROC
[totalScores,index]  = sort(decisionValues);
labels = yTesting;
for i = 1:length(labels)
    labels(i) = yTesting(index(i));
end;


truePositive = zeros(1,length(totalScores)+1);
trueNegative = zeros(1,length(totalScores)+1);
falsePositive = zeros(1,length(totalScores)+1);
falseNegative = zeros(1,length(totalScores)+1);

for i = 1:length(totalScores)
    if labels(i) == 1
        truePositive(1) = truePositive(1)+1;
    else
        falsePositive(1) = falsePositive(1) +1;
    end;
end;

for i = 1:length(totalScores)
   if labels(i) == 1
       truePositive(i+1) = truePositive(i)-1;
       falsePositive(i+1) = falsePositive(i);
   else
       falsePositive(i+1) = falsePositive(i)-1;
       truePositive(i+1) = truePositive(i);
   end;
end;
truePositive = truePositive/truePositive(1);
falsePositive = falsePositive/falsePositive(1);
plot(falsePositive, truePositive);
inc = 0.001;
startIndex = 1;
endIndex = length(falsePositive)
pointerIndex = 1;
pointerValue = falsePositive(1);
newFalsePositive = [];
newTruePositive = [];
while pointerIndex<=length(falsePositive)
    while pointerIndex<=length(falsePositive) && falsePositive(pointerIndex)>falsePositive(startIndex)-inc 
        pointerIndex = pointerIndex +1;
    end;
    newFalsePositive = [newFalsePositive, falsePositive(startIndex)];
    newTruePositive = [newTruePositive, mean(truePositive(startIndex:min(pointerIndex,length(truePositive))))];
    startIndex = pointerIndex;
end;
plot(newFalsePositive, newTruePositive);