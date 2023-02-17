% You should put all your code for recognizing unknown actions in this file.
% Describe the method you used in YourMethod.txt.
% Don't forget to call SavePrediction() at the end with your predicted labels to save them for submission, then submit using submit.m

load("PA9Data.mat");
datasetTrain = datasetTrain3;
datasetTest = datasetTest3;
maxIter = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODIFICATIONS : Initializations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter);
SavePredictions(predicted_labels);

