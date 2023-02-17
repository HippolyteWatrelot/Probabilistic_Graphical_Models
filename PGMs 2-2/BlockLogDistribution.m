%BLOCKLOGDISTRIBUTION
%
%   LogBS = BlockLogDistribution(V, G, F, A) returns the log of a
%   block-sampling array (which contains the log-unnormalized-probabilities of
%   selecting each label for the block), given variables V to block-sample in
%   network G with factors F and current assignment A.  Note that the variables
%   in V must all have the same dimensionality.
%
%   Input arguments:
%   V -- an array of variable names
%   G -- the graph with the following fields:
%     .names - a cell array where names{i} = name of variable i in the graph 
%     .card - an array where card(i) is the cardinality of variable i
%     .edges - a matrix such that edges(i,j) shows if variables i and j 
%              have an edge between them (1 if so, 0 otherwise)
%     .var2factors - a cell array where var2factors{i} gives an array where the
%              entries are the indices of the factors including variable i
%   F -- a struct array of factors.  A factor has the following fields:
%       F(i).var - names of the variables in factor i
%       F(i).card - cardinalities of the variables in factor i
%       F(i).val - a vectorized version of the CPD for factor i (raw probability)
%   A -- an array with 1 entry for each variable in G s.t. A(i) is the current
%       assignment to variable i in G.
%
%   Each entry in LogBS is the log-probability that that value is selected.
%   LogBS is the P(V | X_{-v} = A_{-v}, all X_i in V have the same value), where
%   X_{-v} is the set of variables not in V and A_{-v} is the corresponding
%   assignment to these variables consistent with A.  In the case that |V| = 1,
%   this reduces to Gibbs Sampling.  NOTE that exp(LogBS) is not normalized to
%   sum to one at the end of this function (nor do you need to worry about that
%   in this function).
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function LogBS = BlockLogDistribution(V, G, F, A)
if length(unique(G.card(V))) ~= 1
    disp('WARNING: trying to block sample invalid variable set');
    return;
end

% d is the dimensionality of all the variables we are extracting
d = G.card(V(1));                                                          % Apparently, all variables
                                                                           % have same card
LogBS = zeros(1, d);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% Compute LogBS by multiplying (adding in log-space) in the correct values from
% each factor that includes some variable in V.  
%
% NOTE: As this is called in the innermost loop of both Gibbs and Metropolis-
% Hastings, you should make this fast.  You may want to make use of
% G.var2factors, repmat,unique, and GetValueOfAssignment.
%
% Also you should have only ONE for-loop, as for-loops are VERY slow in matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if length(V) == 1
    factors = G.var2factors{V};
else
    %factors = unique([factors(1:length(factors))]); 
    %disp(factors)
    %disp("lol")
    %factors = unique(factors); 
    %disp(factors)
    %disp("lol2")
    factors = G.var2factors{V(1)};
    for i = 2:length(V)
        factors = union(factors, G.var2factors{V(i)});
    end
    factors = unique(factors);
    %disp(factors)
end
t = transpose(1:d);
for i = 1:length(factors)
    n = factors(i);
    vars = setdiff(F(n).var, V);                                             
    if ~isempty(vars)
        values = A(vars);                                                  
        evidence = cat(2, transpose(vars), transpose(values));
        newfactor = ObserveEvidence(F(n), evidence);                       
        newfactor = FactorMarginalization(newfactor, vars);
    else
        newfactor = F(n);
    end
    newfactor.val = log(newfactor.val);
    ass = repmat(t, 1, length(newfactor.var));
    LogBS = LogBS + GetValueOfAssignment(newfactor, ass);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Re-normalize to prevent underflow when you move back to probability space
LogBS = LogBS - min(LogBS);



