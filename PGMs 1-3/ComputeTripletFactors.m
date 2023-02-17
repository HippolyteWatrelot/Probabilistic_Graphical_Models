function factors = ComputeTripletFactors (images, tripletList, K)
% This function computes the triplet factor values for one word.
%
% Input:
%   images: An array of structs containing the 'img' value for each
%     character in the word.
%   tripletList: An array of the character triplets we will consider (other
%     factor values should be 1). tripletList(i).chars gives character
%     assignment, and triplistList(i).factorVal gives the value for that
%     entry in the factor table.
%   K: The alphabet size (accessible in imageModel.K for the provided
%     imageModel).
%
% Hint: Every character triple in the word will use the same 'val' table.
%   Consider computing that array once and then resusing for each factor.
%
% Copyright (C) Daphne Koller, Stanford University, 2012


n = length(images);

% If the word has fewer than three characters, then return an empty list.
if (n < 3)
    factors = [];
    return
end

factors = repmat(struct('var', [], 'card', [], 'val', []), n - 2, 1);

% Your code here:
allchars1 = transpose(reshape([tripletList(:).chars], [3, length(tripletList)]));
values1 = [tripletList(:).factorVal];
%allchars1 = zeros(length(tripletList), 3);
%values1 = zeros(1, length(tripletList));

for s = 1:n-2
   factors(s).var = [s, s+1, s+2];
   factors(s).card = [K, K, K];
end
factors(1).val = ones(1, prod(factors(1).card));
for j = 1:prod(factors(1).card)
    ind = IndexToAssignment(j, factors(1).card);
    [test, index] = ismember([ind(1), ind(2), ind(3)], allchars1, 'rows');
    if test == 1 
        for r = 1:n-2
            factors(r).val(j) = values1(index);
        end
    end
end
end
