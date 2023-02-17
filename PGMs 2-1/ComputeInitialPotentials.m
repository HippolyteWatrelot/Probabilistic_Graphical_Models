%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
P.edges = zeros(N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
vars = [];
cards = [];
c = 1;
for i = 1:length(C.factorList)
    for j = 1:length(C.factorList(i).var)
        if ismember(C.factorList(i).var(j), vars) == 0
            vars(c) = C.factorList(i).var(j);
            cards(c) = C.factorList(i).card(j);
            c = c + 1;
        end
    end
end
[vars, I] = sort(vars);
cards = cards(I);
for h = 1:N
    clique = cell2mat(C.nodes(h));
    P.cliqueList(h).var = clique;
    for s = 1:length(clique)
        P.cliqueList(h).card(s) = cards(find(vars == clique(s)));
    end
    P.cliqueList(h).val = ones(1, prod(P.cliqueList(h).card));
end

for i = 1:length(C.factorList)
   test = 0;
   for j = 1:N
       clique = cell2mat(C.nodes(j));
       if prod(ismember(C.factorList(i).var, clique)) == 1 
           P.cliqueList(j) = FactorProduct(P.cliqueList(j), C.factorList(i));
           test = 1;
           break;
       end
   end
   if test == 0
       error('FP not respected');
   end
end
for i = 1:N
    P.cliqueList(i) = Reorder(P.cliqueList(i));
    for k = 1:N
        P.edges(i, k) = C.edges(i, k);
    end
end
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% C.nodes is a list of cliques.
% So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end

