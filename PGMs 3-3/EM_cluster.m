% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i = 1:N
      for j = 1:K
          P.c(j) = P.c(j) + ClassProb(i, j) / N;
      end
  end
  for k = 1:K
      for r = 1:size(G, 1)
          X = squeeze(poseData(:, r, :));
          Wk = ClassProb(:, k); 
          if length(size(G)) == 3 
              if G(r, 1, k) == 1 
                  U = squeeze(poseData(:, G(r, 2, k), :));
                  [Beta_yr, sigma_yr] = FitLG(X(:, 1), U, Wk);
                  [Beta_xr, sigma_xr] = FitLG(X(:, 2), U, Wk);
                  [Beta_angler, sigma_angler] = FitLG(X(:, 3), U, Wk);
                  Betas = cat(1, transpose(Beta_yr), transpose(Beta_xr), transpose(Beta_angler));
                  disp(Betas)
                  for indice = 1:numel(Betas)
                      q = fix((indice - 1) / 4) + 1;
                      remain = rem(indice, 4);
                      if remain == 0                                                 % Beta(n+1)
                          P.clg(r).theta(k, indice - 3) = Betas(q, end);
                      else
                          P.clg(r).theta(k, indice + 1) = Betas(q, remain);
                      end
                  end
              else
                  [mu_yr, sigma_yr] = FitG(X(:, 1), Wk);
                  [mu_xr, sigma_xr] = FitG(X(:, 2), Wk);
                  [mu_angler, sigma_angler] = FitG(X(:, 3), Wk);
                  P.clg(r).mu_y(k) = mu_yr;
                  P.clg(r).mu_x(k) = mu_xr;
                  P.clg(r).mu_angle(k) = mu_angler;
              end
          else 
              if G(r, 1) == 1 
                  U = squeeze(poseData(:, G(r, 2), :));
                  [Beta_yr, sigma_yr] = FitLG(X(:, 1), U, Wk);
                  [Beta_xr, sigma_xr] = FitLG(X(:, 2), U, Wk);
                  [Beta_angler, sigma_angler] = FitLG(X(:, 3), U, Wk);
                  Betas = cat(1, transpose(Beta_yr), transpose(Beta_xr), transpose(Beta_angler));
                  for indice = 1:numel(Betas)
                      q = fix((indice - 1) / 4) + 1;
                      remain = rem(indice, 4);
                      if remain == 0                                                 % Beta(n+1)
                          P.clg(r).theta(k, indice - 3) = Betas(q, end);
                      else
                          P.clg(r).theta(k, indice + 1) = Betas(q, remain);
                      end
                  end
              else
                  [mu_yr, sigma_yr] = FitG(X(:, 1), Wk);
                  [mu_xr, sigma_xr] = FitG(X(:, 2), Wk);
                  [mu_angler, sigma_angler] = FitG(X(:, 3), Wk);
                  P.clg(r).mu_y(k) = mu_yr;
                  P.clg(r).mu_x(k) = mu_xr;
                  P.clg(r).mu_angle(k) = mu_angler;
              end
          end
          P.clg(r).sigma_y(k) = sigma_yr;
          P.clg(r).sigma_x(k) = sigma_xr;
          P.clg(r).sigma_angle(k) = sigma_angler;
      end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  ClassProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  tab = zeros(N, K);
  for i = 1:N
      %subterm = zeros(1, K);
      logsubterm = zeros(1, K);
      for k = 1:K
          %bodyterm = zeros(1, size(G, 1));
          logbodyterm = zeros(1, size(G, 1));
          for r = 1:size(G, 1)
              yir = poseData(i, r, 1);
              xir = poseData(i, r, 2);
              angleir = poseData(i, r, 3);
              if length(size(G)) == 3
                  if G(r, 1, k) == 1
                      yipr = poseData(i, G(r, 2, k), 1);
                      xipr = poseData(i, G(r, 2, k), 2);
                      alphaipr = poseData(i, G(r, 2, k), 3);
                      mu_yrk = sum([1, yipr, xipr, alphaipr] .* P.clg(r).theta(k, 1:4));
                      mu_xrk = sum([1, yipr, xipr, alphaipr] .* P.clg(r).theta(k, 5:8));
                      mu_anglerk = sum([1, yipr, xipr, alphaipr] .* P.clg(r).theta(k, 9:12));
                  else
                      mu_yrk = P.clg(r).mu_y(k);
                      mu_xrk = P.clg(r).mu_x(k);
                      mu_anglerk = P.clg(r).mu_angle(k);
                  end
              else
                  if G(r, 1) == 1
                      yipr = poseData(i, G(r, 2), 1);
                      xipr = poseData(i, G(r, 2), 2);
                      alphaipr = poseData(i, G(r, 2), 3);
                      mu_yrk = sum([1, yipr, xipr, alphaipr] .* P.clg(r).theta(k, 1:4));
                      mu_xrk = sum([1, yipr, xipr, alphaipr] .* P.clg(r).theta(k, 5:8));
                      mu_anglerk = sum([1, yipr, xipr, alphaipr] .* P.clg(r).theta(k, 9:12));
                  else
                      mu_yrk = P.clg(r).mu_y(k);
                      mu_xrk = P.clg(r).mu_x(k);
                      mu_anglerk = P.clg(r).mu_angle(k);
                  end
              end
              sigma_yrk = P.clg(r).sigma_y(k);
              sigma_xrk = P.clg(r).sigma_x(k);
              sigma_anglerk = P.clg(r).sigma_angle(k);
              logProbyrk = lognormpdf(yir, mu_yrk, sigma_yrk);
              logProbxrk = lognormpdf(xir, mu_xrk, sigma_xrk);
              logProbanglerk = lognormpdf(angleir, mu_anglerk, sigma_anglerk);
              logProbClassInference = logProbyrk + logProbxrk + logProbanglerk;
              logbodyterm(r) = logProbClassInference;
          end
          logsubterm(k) = sum(logbodyterm) + log(P.c(k));
      end
      logProbInstance = logsumexp(logsubterm);
      for j = 1:K
          logProbClassInstance = logsubterm(j);
          tab(i, j) = logProbClassInstance;
          ClassProb(i, j) = exp(logProbClassInstance - logProbInstance);
      end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of dataset for this iteration
  % Hint: You should use the logsumexp() function here
  loglikelihood(iter) = 0;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  loglikelihood(iter) = sum(logsumexp(tab));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
