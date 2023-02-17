% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over X_2 and X_3, which takes on a value of 1
    % if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);                      % X done here
    % featureSet is relative to X, We can have same singleton feature with
    % same assignment but different paramIdx --> One of the two is actually a X-conditional feature.
    % numParams is for theta

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.
    
    nll = 0;
    grad = zeros(size(theta));
    %%%
    % Your code here:
    % F must come from featureset and theta
    % Build factors thanks to prob exponential formula
    N = length(featureSet.features);
    K = modelParams.numHiddenStates;
    numfactors = 2 * length(y) - 1;      % Singletons (fusioned with X-conditionals) + Pairwises
    F = repmat(struct('var', [], 'card', [], 'val', []), numfactors, 1);

    for i = 1:numfactors
        if rem(i, 2) == 1
            F(i).var = 1 + (i-1)/2;
            F(i).card = K;
            F(i).val = ones(1, K);
        else
            F(i).var = [i/2, i/2 + 1];
            F(i).card = K * ones(1, 2);
            F(i).val = ones(1, K^2);
        end
    end
 
    underterms = zeros(1, length(theta));                                  % For data expectancy
    repartition = repmat(struct("features", []), length(theta));
    features_tests = zeros(1, N);
    for i = 1:N
        ass_i = featureSet.features(i).assignment;
        current_theta = theta(featureSet.features(i).paramIdx);
        current_param = featureSet.features(i).paramIdx;
        repartition(current_param).features(end + 1) = i;
        if length(featureSet.features(i).var) == 1
            n = featureSet.features(i).var;
            indice = 2*n-1;
            if y(featureSet.features(i).var) == ass_i
                current_theta = theta(featureSet.features(i).paramIdx);
                features_tests(i) = 1;
                underterms(current_param) = underterms(current_param) + 1;
            end
        else
            n = featureSet.features(i).var(1);
            indice = 2*n;
            if y(featureSet.features(i).var(1)) == ass_i(1) && y(featureSet.features(i).var(2)) == ass_i(2)
                current_theta = theta(current_param);
                features_tests(i) = 1;
                underterms(current_param) = underterms(current_param) + 1;
            end
        end
        ind = AssignmentToIndex(ass_i, F(indice).card);
        F(indice).val(ind) = F(indice).val(ind) * exp(current_theta);
    end
    %disp(F)
    P = CreateCliqueTree(F);
    [P, logZ] = CliqueTreeCalibrate(P, 0);
    features_counts = underterms;
    weighted_features_counts = theta.*features_counts;
    nll = logZ - sum(weighted_features_counts) + (modelParams.lambda / 2) * sum(theta.^2);
    model_features_counts = zeros(1, length(theta));                       % E0[fi]

    PS1 = repmat(struct('var', 0, 'card', 0, 'val', []), length(y));
    PS2 = repmat(struct('var', 0, 'card', 0, 'val', []), length(y) - 1);
    for c = 1:length(y) - 1
        PS1(c) = ComputeMarginal(P.cliqueList(c).var(1), P.cliqueList(c), []);
        PS2(c) = P.cliqueList(c);
        PS2(c).val = PS2(c).val / sum(PS2(c).val);
    end
    PS1(length(y)) = ComputeMarginal(P.cliqueList(end).var(2), P.cliqueList(end), []);

    for i = 1:N
        current_theta = featureSet.features(i).paramIdx;
        if length(featureSet.features(i).var) == 1
            factor = PS1(featureSet.features(i).var);
            ind = featureSet.features(i).assignment;
        else
            factor = PS2(featureSet.features(i).var(1));
            ind = AssignmentToIndex(featureSet.features(i).assignment, K*ones(1, 2));
        end
        p = factor.val(ind);
        model_features_counts(current_theta) = model_features_counts(current_theta) + p; 
    end

    for i = 1:length(theta)
        grad(i) = model_features_counts(i) - features_counts(i) + modelParams.lambda * theta(i);
    end
end
