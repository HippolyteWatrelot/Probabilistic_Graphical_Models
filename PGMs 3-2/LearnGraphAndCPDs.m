function [P G loglikelihood] = LearnGraphAndCPDs(dataset, labels)

% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha) 
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

G = zeros(10,2,K); % graph structures to learn
% initialization
for k=1:K
    G(2:end,:,k) = ones(9,2);
end

% estimate graph structure for each class
% YOUR CODE HERE
for k=1:K
    % fill in G(:,:,k)
    % use ConvertAtoG to convert a maximum spanning tree to a graph G
    P.c(k) = sum(labels(:, k)) / N;
    indk = transpose(find(labels(:, k)));
    dataset_k = dataset(indk, :, :);
    [A, W] = LearnGraphStructure(dataset_k);
    G(:, :, k) = ConvertAtoG(A);
    for r = 1:size(G, 1)
        X = squeeze(dataset_k(:, r, :));                      
        if G(r, 1, k) == 1                                        % Variable has parent
            U = squeeze(dataset_k(:, G(r, 2, k), :));
            [Beta_yr, sigma_yr] = FitLinearGaussianParameters(X(:, 1), U);
            [Beta_xr, sigma_xr] = FitLinearGaussianParameters(X(:, 2), U);
            [Beta_angler, sigma_angler] = FitLinearGaussianParameters(X(:, 3), U);
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
        else                                                                   % Variable doesn't have a parent
            [mu_yr, sigma_yr] = FitGaussianParameters(X(:, 1));
            [mu_xr, sigma_xr] = FitGaussianParameters(X(:, 2));
            [mu_angler, sigma_angler] = FitGaussianParameters(X(:, 3));
            P.clg(r).mu_y(k) = mu_yr;
            P.clg(r).mu_x(k) = mu_xr;
            P.clg(r).mu_angle(k) = mu_angler;
        end
        P.clg(r).sigma_y(k) = sigma_yr;
        P.clg(r).sigma_x(k) = sigma_xr;
        P.clg(r).sigma_angle(k) = sigma_angler;
    end
end
loglikelihood = ComputeLogLikelihood(P, G, dataset);

% estimate parameters

%%%%%%%%P.c = zeros(1,K);
% compute P.c
% the following code can be copied from LearnCPDsGivenGraph.m
% with little or no modification
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE

% These are dummy lines added so that submit.m will run even if you 
% have not started coding. Please delete them.
%P.clg.sigma_x = 0;
%P.clg.sigma_y = 0;
%P.clg.sigma_angle = 0;
%loglikelihood = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('log likelihood: %f\n', loglikelihood);