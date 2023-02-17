function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels, 2);
accuracy = 0.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
for j = 1:N
    true_label = find(labels(j, :));
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
        subterm(k) = prod(bodyterm) * P.c(k);                              % config&class probabilities 
    end
    prob = sum(subterm);                                                   % Unconditional probability of config
    tests = zeros(1, 2);                                                   % classes probabilities knowing config
    for k = 1:K
        tests(k) = subterm(k) / prob;
    end
    [~, test] = max(tests);
    if test == true_label
        accuracy = accuracy + 1 / N;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Accuracy: %.2f\n', accuracy);