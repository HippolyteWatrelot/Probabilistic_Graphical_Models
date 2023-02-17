%GETNEXTCLIQUES Find a pair of cliques ready for message passing
%   [i, j] = GETNEXTCLIQUES(P, messages) finds ready cliques in a given
%   clique tree, P, and a matrix of current messages. Returns indices i and j
%   such that clique i is ready to transmit a message to clique j.
%
%   We are doing clique tree message passing, so
%   do not return (i,j) if clique i has already passed a message to clique j.
%
%	 messages is a n x n matrix of passed messages, where messages(i,j)
% 	 represents the message going from clique i to clique j. 
%   This matrix is initialized in CliqueTreeCalibrate as such:
%      MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
%
%   If more than one message is ready to be transmitted, return 
%   the pair (i,j) that is numerically smallest. If you use an outer
%   for loop over i and an inner for loop over j, breaking when you find a 
%   ready pair of cliques, you will get the right answer.
%
%   If no such cliques exist, returns i = j = 0.
%
%   See also CLIQUETREECALIBRATE
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function [i, j] = GetNextCliques(P, messages)

% initialization
% you should set them to the correct values in your code
i = 0;
j = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
N = length(messages);
test = 0;
%core = 8; 
for receiver = 1:N 
    %disp(receiver)
    counter = 0;
    memory = 0;
    downward = 0;
    for giver = 1:N 
        if P.edges(giver, receiver) == 1 && isempty(messages(giver, receiver).var) 
            counter = counter + 1;
            memory = giver;
            test = 1;
        end
        if counter == 2
            test = 0;
            break;
        end
        if giver == N && counter == 0
            test = 1;
            downward = 1;                                                   % We are in downward mode
        end
    end
    if test == 1
        i = receiver;                                                        % Which is now a giver
        if memory ~= 0 && isempty(messages(i, memory).var) %&& i ~= core     % When i received all from its neighbors but one
            j = memory;
            return;
        else
            for k = 1:N
                if P.edges(i, k) == 1 && isempty(messages(i, k).var) && downward == 1
                    j = k;
                    return;
                end
            end
        end
        if j == 0
            i = 0;
        end
        if receiver == N
            return;
        end
        test = 0;
    end
    if j ~= 0
        break;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



return;
