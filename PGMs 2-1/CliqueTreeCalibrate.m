%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = 2*(N - 1);                                                              % Upward & Downward
for s = 1:K
    [i, j] = GetNextCliques(P, MESSAGES);
    %disp([i, j])
    term = P.cliqueList(i);
    if isMax == 1
        term.val = log(term.val); 
    end
    for k = 1:N
        if isMax == 0 
            if k ~= j
                term = FactorProduct(term, MESSAGES(k, i));  
                term.val = term.val / sum(term.val);
            end
            if isempty(MESSAGES(k, i).var) == 0
                %disp([0, 0, 0, k, i])
                %disp(MESSAGES(k, i))
            end
        elseif isMax == 1 
            if k ~= j
                term = FactorSum(term, MESSAGES(k, i));
            end
        else
            error("isMax must be 0 or 1")
        end
    end
    %disp(P.cliqueList(i))
    %disp(term)
    sumfactors = [];
    counter = 1;
    for r = 1:length(term.var)
        if ismember(term.var(r), P.cliqueList(j).var) == 0
            sumfactors(counter) = term.var(r);
            counter = counter + 1;
        end
    end
    %disp(sumfactors)
    if isMax == 0
        MSG = FactorMarginalization(term, sumfactors);
        MSG.val = MSG.val / sum(MSG.val);
    else
        MSG = FactorMaxMarginalization(term, sumfactors);
    end
    MESSAGES(i, j) = MSG;
    %disp(MESSAGES(i, j))
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
for i = 1:N
    if isMax == 1
        P.cliqueList(i).val = log(P.cliqueList(i).val);
    end
    disp(P.cliqueList(i))
    for j = 1:N
        if isMax == 0 
            P.cliqueList(i) = FactorProduct(P.cliqueList(i), MESSAGES(j, i));
        else
            P.cliqueList(i) = FactorSum(P.cliqueList(i), MESSAGES(j, i));
        end
    end
end
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



return
