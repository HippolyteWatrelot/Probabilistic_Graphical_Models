function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% For one body part i of one j sample,
% P(Oi | C = k, Opi) = P(yi | C = k, Opi) * P(xi | C = k, Opi) * P(alphai | C = k, Opi) 
% Extracting yji, xji and alphaji in dataset
% Use class k (same for all body parts)
% Building mu and sigma yji, xji and alphaji
% P(Oi | C = k, Opi) = exp(lognormpdf(yji, mu_yji, sigma_yji) 
% + lognormpdf(xji, mu_xji, sigma_xji) + lognormpdf(alphaji, mu_alphaji, sigma_alphaji))
for j = 1:N
    subterm = zeros(1, K);
    for k = 1:K
        bodyterm = zeros(1, size(G, 1));
        for i = 1:size(G, 1)
            yji = dataset(j, i, 1);
            xji = dataset(j, i, 2);
            alphaji = dataset(j, i, 3);
            if length(size(G)) == 3
                if G(i, 1, k) == 1
                    yjpi = dataset(j, G(i, 2, k), 1);
                    xjpi = dataset(j, G(i, 2, k), 2);
                    alphajpi = dataset(j, G(i, 2, k), 3);
                    mu_yji = sum([1, yjpi, xjpi, alphajpi] .* P.clg(i).theta(k, 1:4));
                    mu_xji = sum([1, yjpi, xjpi, alphajpi] .* P.clg(i).theta(k, 5:8));
                    mu_alphaji = sum([1, yjpi, xjpi, alphajpi] .* P.clg(i).theta(k, 9:12));
                else
                    mu_yji = P.clg(i).mu_y(k);
                    mu_xji = P.clg(i).mu_x(k);
                    mu_alphaji = P.clg(i).mu_angle(k);
                end
            else
                if G(i, 1) == 1
                    yjpi = dataset(j, G(i, 2), 1);
                    xjpi = dataset(j, G(i, 2), 2);
                    alphajpi = dataset(j, G(i, 2), 3);
                    mu_yji = sum([1, yjpi, xjpi, alphajpi] .* P.clg(i).theta(k, 1:4));
                    mu_xji = sum([1, yjpi, xjpi, alphajpi] .* P.clg(i).theta(k, 5:8));
                    mu_alphaji = sum([1, yjpi, xjpi, alphajpi] .* P.clg(i).theta(k, 9:12));
                else
                    mu_yji = P.clg(i).mu_y(k);
                    mu_xji = P.clg(i).mu_x(k);
                    mu_alphaji = P.clg(i).mu_angle(k);
                end
            end
            sigma_yji = P.clg(i).sigma_y(k);
            sigma_xji = P.clg(i).sigma_x(k);
            sigma_alphaji = P.clg(i).sigma_angle(k);
            logProb_y = lognormpdf(yji, mu_yji, sigma_yji);
            logProb_x = lognormpdf(xji, mu_xji, sigma_xji);
            logProb_alpha = lognormpdf(alphaji, mu_alphaji, sigma_alphaji);
            logProb = logProb_y + logProb_x + logProb_alpha;
            bodyterm(i) = exp(logProb);
        end
        subterm(k) = prod(bodyterm) * P.c(k);
    end
    loglikelihood = loglikelihood + log(sum(subterm));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
