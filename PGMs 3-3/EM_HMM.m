% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  for i = 1:L
      init_ind = actionData(i).marg_ind(1);
      for j = 1:K
          P.c(j) = P.c(j) + ClassProb(init_ind, j) / L;
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
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);
  
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i = 1:L
      for k = 1:length(actionData(i).pair_ind)
          pair_indice = actionData(i).pair_ind(k);
          test = reshape(PairProb(pair_indice, :), K, K);
          for j = 1:K
              for h = 1:K
                  %P.transMatrix(j, h) = P.transMatrix(j, h) + PairProb(pair_indice, (j-1)*K + h);
                  P.transMatrix(j, h) = P.transMatrix(j, h) + test(j, h);
              end
          end
      end
  end
  for j = 1:K
      %P.transMatrix(j, :) = P.transMatrix(j, :) / P.c(j);                     % P(B&A) -> P(B|A) (Useless)
      P.transMatrix(j, :) = P.transMatrix(j, :) / sum(P.transMatrix(j, :));    % Normalization
  end
  % P is now "max likehooded" for the data with the given ClassProb and PairProb
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %tab = zeros(N, K);
  for i = 1:N
      logsubterm = zeros(1, K);
      for k = 1:K
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
          logsubterm(k) = sum(logbodyterm);                                % Specific class
      end
      for j = 1:K
          logProbInstanceKnowingClass = logsubterm(j);
          logEmissionProb(i, j) = logProbInstanceKnowingClass;             % To be used in factors
      end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);                                                  
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %tab = zeros(N, K);
  TAB = zeros(L, 1);
  for i = 1:L                       % Number of (same) actions 
      factors = repmat(struct('var', [], 'card', [], 'val', []), length(actionData(i).pair_ind) + length(actionData(i).marg_ind) + 1, 1);
      for j = 1:length(actionData(i).marg_ind)                          % Pose probs in action
          factors(j).var = j;
          factors(j).card = K;
          factors(j).val = logEmissionProb(actionData(i).marg_ind(j), :);
      end
      factors(length(actionData(i).marg_ind) + 1).var = 1;              % First pose in action
      factors(length(actionData(i).marg_ind) + 1).card = K;
      factors(length(actionData(i).marg_ind) + 1).val = log(P.c);
      for j = 1:length(actionData(i).pair_ind)                          % Next poses (Knowing previous one) in action
          factors(length(actionData(i).marg_ind) + 1 + j).var = [j+1, j];
          factors(length(actionData(i).marg_ind) + 1 + j).card = [K, K];
          factors(length(actionData(i).marg_ind) + 1 + j).val = reshape(transpose(log(P.transMatrix)), 1, K^2);      % Must be in log form
      end
      [M, Tree] = ComputeExactMarginalsHMM(factors);
      for r = 1:length(M) - 1                                                  % (length(M) - 1 = pair_ind length)
          dual_factor = Tree.cliqueList(r);
          mat = zeros(K, K);
          for k = 1:K
              for j = 1:K
                  ind = AssignmentToIndex([k, j], dual_factor.card);
                  logProb_srplus1jANDsrk = dual_factor.val(ind) - logsumexp(dual_factor.val);      % Normalization
                  mat(k, j) = exp(logProb_srplus1jANDsrk);
                  % --> Pi(Sr+1=j, Sr=k | ai)
              end
          end
          PairProb(actionData(i).pair_ind(r), :) = reshape(mat, 1, K^2);
      end
      for r = 1:length(M)                                                  % (length(M) = marg_ind length)
          factor = M(r);
          %factor = Tree.cliqueList(r);
          for k = 1:K
              logProbsr = factor.val(k);
              %f = FactorMarginalization(factor, factor.var(2));
              %logProbsr = f.val(k) - logsumexp(f.val);
              ClassProb(actionData(i).marg_ind(r), k) = exp(logProbsr);
              % --> Pi(Sr=k | ai)
              %tab(actionData(i).marg_ind(r), k) = logEmissionProb(actionData(i).marg_ind(r), k) + log(P.c(k));
          end
      end
      %factor = Tree.cliqueList(end);
      %for k = 1:K
      %    f = FactorMarginalization(factor, factor.var(1));
      %    logProbsr = f.val(k) - logsumexp(f.val);
      %    ClassProb(actionData(i).marg_ind(end), k) = exp(logProbsr);
      %end
      TAB(i) = logsumexp(Tree.cliqueList(1).val);
  end

  %  LIKELIHOOD

  %loglikelihood(iter) = sum(logsumexp(tab));
  % loglikelihood(iter) should be a sum over actions
  %loglikelihood(iter) = sum(logsumexp(log(ClassProb))) + sum(logsumexp(log(PairProb))) + sum(logsumexp(logEmissionProb));
  loglikelihood(iter) = sum(TAB);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
