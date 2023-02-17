% File: RecognizeActions.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter)

% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ClapModel, ll1, CP1, PB1] = EM_HMM(datasetTrain(1).actionData, datasetTrain(1).poseData, G, datasetTrain(1).InitialClassProb, datasetTrain(1).InitialPairProb, maxIter);
[HKModel, ll2, CP2, PB2] = EM_HMM(datasetTrain(2).actionData, datasetTrain(2).poseData, G, datasetTrain(2).InitialClassProb, datasetTrain(2).InitialPairProb, maxIter);
[LKModel, ll3, CP3, PB3] = EM_HMM(datasetTrain(3).actionData, datasetTrain(3).poseData, G, datasetTrain(3).InitialClassProb, datasetTrain(3).InitialPairProb, maxIter);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Classify each of the instances in datasetTest
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

accuracy = 0;
%predicted_labels = [];
predicted_labels = zeros(size(datasetTest.actionData, 2), 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ps(1) = ClapModel;
Ps(2) = HKModel;
Ps(3) = LKModel;
%likelihoods = zeros(1, 3);
%CPs = zeros(3, N, K);
%PPs = zeros(3, V, K^2);
%N = size(dataTest.poseData, 1);
L = size(datasetTest.actionData, 2);
K = size(datasetTrain(1).InitialClassProb, 2);

for action = 1:L
    Nposes = length(datasetTest.actionData(action).marg_ind);
    logEmissionProbs = zeros(3, Nposes, K);
    if isfield(datasetTest, 'labels')
        label = datasetTest.labels(action);
    end
    for i = 1:Nposes
        indice = datasetTest.actionData(action).marg_ind(i);
        for model = 1:3
            logsubterm = zeros(1, K);
            for k = 1:K
                logbodyterm = zeros(1, size(G, 1));
                for r = 1:size(G, 1)
                    yir = datasetTest.poseData(indice, r, 1);
                    xir = datasetTest.poseData(indice, r, 2);
                    angleir = datasetTest.poseData(indice, r, 3);
                    if length(size(G)) == 3
                        if G(r, 1, k) == 1
                            yipr = datasetTest.poseData(indice, G(r, 2, k), 1);
                            xipr = datasetTest.poseData(indice, G(r, 2, k), 2);
                            alphaipr = datasetTest.poseData(indice, G(r, 2, k), 3);
                            mu_yrk = sum([1, yipr, xipr, alphaipr] .* Ps(model).clg(r).theta(k, 1:4));
                            mu_xrk = sum([1, yipr, xipr, alphaipr] .* Ps(model).clg(r).theta(k, 5:8));
                            mu_anglerk = sum([1, yipr, xipr, alphaipr] .* Ps(model).clg(r).theta(k, 9:12));
                        else
                            mu_yrk = Ps(model).clg(r).mu_y(k);
                            mu_xrk = Ps(model).clg(r).mu_x(k);
                            mu_anglerk = Ps(model).clg(r).mu_angle(k);
                        end              
                    else
                        if G(r, 1) == 1
                            yipr = datasetTest.poseData(indice, G(r, 2), 1);
                            xipr = datasetTest.poseData(indice, G(r, 2), 2);
                            alphaipr = datasetTest.poseData(indice, G(r, 2), 3);
                            mu_yrk = sum([1, yipr, xipr, alphaipr] .* Ps(model).clg(r).theta(k, 1:4));
                            mu_xrk = sum([1, yipr, xipr, alphaipr] .* Ps(model).clg(r).theta(k, 5:8));
                            mu_anglerk = sum([1, yipr, xipr, alphaipr] .* Ps(model).clg(r).theta(k, 9:12));
                        else
                            mu_yrk = Ps(model).clg(r).mu_y(k);
                            mu_xrk = Ps(model).clg(r).mu_x(k);
                            mu_anglerk = Ps(model).clg(r).mu_angle(k);
                        end
                    end
                    sigma_yrk = Ps(model).clg(r).sigma_y(k);
                    sigma_xrk = Ps(model).clg(r).sigma_x(k);
                    sigma_anglerk = Ps(model).clg(r).sigma_angle(k);
                    logProbyrk = lognormpdf(yir, mu_yrk, sigma_yrk);
                    logProbxrk = lognormpdf(xir, mu_xrk, sigma_xrk);
                    logProbanglerk = lognormpdf(angleir, mu_anglerk, sigma_anglerk);
                    logProbClassInference = logProbyrk + logProbxrk + logProbanglerk;
                    logbodyterm(r) = logProbClassInference;
                end
                logsubterm(k) = sum(logbodyterm);                                % Specific class
            end
            for j = 1:K
                logProbInstanceKnowingClass = logsubterm(j);
                logEmissionProbs(model, i, j) = logProbInstanceKnowingClass;             % To be used in factors
            end
        end
    end
    
    Clapfactors = repmat(struct('var', [], 'card', [], 'val', []), 2*Nposes, 1);
    HKfactors = repmat(struct('var', [], 'card', [], 'val', []), 2*Nposes, 1);
    LKfactors = repmat(struct('var', [], 'card', [], 'val', []), 2*Nposes, 1);
    for m = 1:3
        factors = repmat(struct('var', [], 'card', [], 'val', []), 2*Nposes, 1);
        for j = 1:length(datasetTest.actionData(action).marg_ind)                          
            factors(j).var = j;
            factors(j).card = K;
            factors(j).val = reshape((logEmissionProbs(m, j, :)), 1, K);
        end
        factors(length(datasetTest.actionData(action).marg_ind) + 1).var = 1;              
        factors(length(datasetTest.actionData(action).marg_ind) + 1).card = K;
        factors(length(datasetTest.actionData(action).marg_ind) + 1).val = reshape((log(Ps(m).c)), 1, K);
        for j = 1:length(datasetTest.actionData(action).pair_ind)                          
            factors(length(datasetTest.actionData(action).marg_ind) + 1 + j).var = [j+1, j];
            factors(length(datasetTest.actionData(action).marg_ind) + 1 + j).card = [K, K];
            factors(length(datasetTest.actionData(action).marg_ind) + 1 + j).val = reshape(transpose(log(Ps(m).transMatrix)), 1, K^2);      
        end
        if m == 1
            Clapfactors = factors;
        elseif m == 2
            HKfactors = factors;
        else
            LKfactors = factors;
        end
    end
    [~, Tree1] = ComputeExactMarginalsHMM(Clapfactors);
    [~, Tree2] = ComputeExactMarginalsHMM(HKfactors);
    [~, Tree3] = ComputeExactMarginalsHMM(LKfactors);

    test_likelihoods = zeros(1, 3);
    test_likelihoods(1) = logsumexp(Tree1.cliqueList(1).val);
    test_likelihoods(2) = logsumexp(Tree2.cliqueList(1).val);
    test_likelihoods(3) = logsumexp(Tree3.cliqueList(1).val);
    [~, test] = max(test_likelihoods);
    if isfield(datasetTest, 'labels')
        accuracy = accuracy + (test == label) / L;
    end
    predicted_labels(action) = test;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
