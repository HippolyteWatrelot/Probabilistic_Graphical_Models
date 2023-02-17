function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
ind1 = transpose(find(labels(:, 1)));
ind2 = transpose(find(labels(:, 2)));
dataset_1 = dataset(ind1, :, :);                 % Human Dataset 
dataset_2 = dataset(ind2, :, :);                 % Alien Dataset
for i = 1:N
    for j = 1:K
        P.c(j) = P.c(j) + labels(i, j) / N;
    end
end
disp(size(G))
for r = 1:size(G, 1)
    X1 = squeeze(dataset_1(:, r, :));                      % Human BodyPart(r) dataset
    X2 = squeeze(dataset_2(:, r, :));                      % Alien BodyPart(r) dataset
    if G(r, 1) == 1                                        % Variable has parent
        U1 = squeeze(dataset_1(:, G(r, 2), :));
        U2 = squeeze(dataset_2(:, G(r, 2), :));
        [Beta_yr1, sigma_yr1] = FitLinearGaussianParameters(X1(:, 1), U1);
        [Beta_xr1, sigma_xr1] = FitLinearGaussianParameters(X1(:, 2), U1);
        [Beta_angler1, sigma_angler1] = FitLinearGaussianParameters(X1(:, 3), U1);
        [Beta_yr2, sigma_yr2] = FitLinearGaussianParameters(X2(:, 1), U2);
        [Beta_xr2, sigma_xr2] = FitLinearGaussianParameters(X2(:, 2), U2);
        [Beta_angler2, sigma_angler2] = FitLinearGaussianParameters(X2(:, 3), U2);
        Betas_1 = cat(1, transpose(Beta_yr1), transpose(Beta_xr1), transpose(Beta_angler1));
        Betas_2 = cat(1, transpose(Beta_yr2), transpose(Beta_xr2), transpose(Beta_angler2));
        for indice = 1:numel(Betas_1)
            k = fix((indice - 1) / 4) + 1;
            remain = rem(indice, 4);
            if remain == 0                                                 % Beta(n+1)
                P.clg(r).theta(1, indice - 3) = Betas_1(k, end);
                P.clg(r).theta(2, indice - 3) = Betas_2(k, end);
            else
                P.clg(r).theta(1, indice + 1) = Betas_1(k, remain);
                P.clg(r).theta(2, indice + 1) = Betas_2(k, remain);
            end
        end
    else                                                                   % Variable doesn't have a parent
        [mu_yr1, sigma_yr1] = FitGaussianParameters(X1(:, 1));
        [mu_xr1, sigma_xr1] = FitGaussianParameters(X1(:, 2));
        [mu_angler1, sigma_angler1] = FitGaussianParameters(X1(:, 3));
        [mu_yr2, sigma_yr2] = FitGaussianParameters(X2(:, 1));
        [mu_xr2, sigma_xr2] = FitGaussianParameters(X2(:, 2));
        [mu_angler2, sigma_angler2] = FitGaussianParameters(X2(:, 3));
        P.clg(r).mu_y(1) = mu_yr1;
        P.clg(r).mu_x(1) = mu_xr1;
        P.clg(r).mu_angle(1) = mu_angler1;
        P.clg(r).mu_y(2) = mu_yr2;
        P.clg(r).mu_x(2) = mu_xr2;
        P.clg(r).mu_angle(2) = mu_angler2;
    end
    P.clg(r).sigma_y(1) = sigma_yr1;
    P.clg(r).sigma_x(1) = sigma_xr1;
    P.clg(r).sigma_angle(1) = sigma_angler1;
    P.clg(r).sigma_y(2) = sigma_yr2;
    P.clg(r).sigma_x(2) = sigma_xr2;
    P.clg(r).sigma_angle(2) = sigma_angler2;
end
loglikelihood = ComputeLogLikelihood(P, G, dataset);
% These are dummy lines added so that submit.m will run even if you 
% have not started coding. Please delete them.
%P.clg.sigma_x = 0;
%P.clg.sigma_y = 0;
%P.clg.sigma_angle = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('log likelihood: %f\n', loglikelihood);

