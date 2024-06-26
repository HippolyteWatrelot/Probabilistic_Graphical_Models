%ComputeMarginal Computes the marginal over a set of given variables
%   M = ComputeMarginal(V, F, E) computes the marginal over variables V
%   in the distribution induced by the set of factors F, given evidence E
%
%   M is a factor containing the marginal over variables V
%   V is a vector containing the variables in the marginal e.g. [1 2 3] for
%     X_1, X_2 and X_3.
%   F is a vector of factors (struct array) containing the factors 
%     defining the distribution
%   E is an N-by-2 matrix, each row being a variable/value pair. 
%     Variables are in the first column and values are in the second column.
%     If there is no evidence, pass in the empty matrix [] for E.


function M = ComputeMarginal(V, F, E)

% Check for empty factor list
if (numel(F) == 0)
      warning('Warning: empty factor list');
      M = struct('var', [], 'card', [], 'val', []);      
      return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE:
% M should be a factor
% Remember to renormalize the entries of M!
if isempty(E)
    evidences = F;
else
    evidences = ObserveEvidence(F, E);
end
evidences_joint = ComputeJointDistribution(evidences);
var = evidences_joint.var;
for i = 1:length(evidences_joint.var)
    for j = 1:length(V)
        if V(j) == evidences_joint.var(i)
            var = var(var~=V(j));
        end
    end
end
margin_evidences_joint = FactorMarginalization(evidences_joint, var);
card = margin_evidences_joint.card;
val = margin_evidences_joint.val / sum(margin_evidences_joint.val);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = struct('var', V, 'card', card, 'val', val); % Returns empty factor. Change this.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
