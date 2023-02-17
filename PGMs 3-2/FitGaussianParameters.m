function [mu sigma] = FitGaussianParameters(X)
% X: (N x 1): N examples (1 dimensional)
% Fit N(mu, sigma^2) to the empirical distribution
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

mu = 0;
sigma = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
N = length(X);
mu = sum(X) / N;
EX2 = sum(X.*X) / N;
sigma = sqrt(EX2 - mu^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%