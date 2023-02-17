%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

% initialization
% you should set it to the correct value in your code
M = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
P = CreateCliqueTree(F, E);
finalP = CliqueTreeCalibrate(P, isMax);
CL = finalP.cliqueList;
tab = [];
counter = 1;
for i = 1:length(CL)
    for j = 1:length(CL(i).var)
        if ismember(CL(i).var(j), tab) == 0
            tab(counter) = CL(i).var(j);
            counter = counter + 1;
        end
    end
end
tab = sort(tab);                                                           % all vars
M = repmat(struct('var', [], 'card', [], 'val', []), length(tab), 1);
for i = 1:length(tab)
    for j = 1:length(CL)
        if ismember(tab(i), CL(j).var)
            V = setdiff(CL(j).var, tab(i));
            if isMax == 0
                M(i) = FactorMarginalization(CL(j), V);
                M(i).val = M(i).val / sum(M(i).val);
            else 
                M(i) = FactorMaxMarginalization(CL(j), V);
            end
            break;
        end
    end
end
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
