% FactorMarginalization Sums given variables out of a factor.
%   B = FactorMarginalization(A,V) computes the factor with the variables
%   in V summed out. The factor data structure has the following fields:
%       .var    Vector of variables in the factor, e.g. [1 2 3]
%       .card   Vector of cardinalities corresponding to .var, e.g. [2 2 2]
%       .val    Value table of size prod(.card)
%
%   The resultant factor should have at least one variable remaining or this
%   function will throw an error.
% 
%   See also FactorProduct.m, IndexToAssignment.m, and AssignmentToIndex.m

function B = FactorMarginalization(A, V)

% Check for empty factor or variable list
if (isempty(A.var) || isempty(V)), B = A; return; end;

% Construct the output factor over A.var \ V (the variables in A.var that are not in V)
% and mapping between variables in A and B
[B.var, mapB] = setdiff(A.var, V);

% Check for empty resultant factor
if isempty(B.var)
  error('Error: Resultant factor has empty scope');
end;

% Initialize B.card and B.val
B.card = A.card(mapB);
B.val = zeros(1, prod(B.card));

% Compute some helper indices
% These will be very useful for calculating B.val
% so make sure you understand what these lines are doing
assignments = IndexToAssignment(1:length(A.val), A.card);
indxB = AssignmentToIndex(assignments(:, mapB), B.card);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% Correctly populate the factor values of B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
resh_valA = reshape(A.val, A.card);
new_varA = A.var;
new_cardA = A.card;
indice = 0;
for j = 1:length(A.var)
    for i = 1:length(V)
        if V(i) == A.var(j)
            resh_valA = sum(resh_valA, j-indice);
            indice = indice + 1;
            new_cardA = new_cardA(new_varA~=V(i));
            new_varA = new_varA(new_varA~=V(i));
        end
    end
end
if length(new_cardA) >= 2
    resh_valA = reshape(resh_valA, [1, prod(new_cardA)]);
end
for i = 1:length(resh_valA)
    B.val(i) = resh_valA(i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
