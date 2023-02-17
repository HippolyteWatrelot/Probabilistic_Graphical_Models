function F = FactorSum(A,B)
F = struct('var', [], 'card', [], 'val', []);
F.var = A.var;
F.card = A.card;
index = length(F.var) + 1;
for i = 1:length(B.var)
    if ismember(B.var(i), F.var) == 0
        F.var(index) = B.var(i);
        F.card(index) = B.card(i);
        index = index + 1;
    end
end
F.val = zeros(1, prod(F.card));
for k = 1:prod(F.card)
    ind = IndexToAssignment(k, F.card);
    Aval = A.val(AssignmentToIndex(ind(1:length(A.var)), A.card));
    indB = zeros(1, length(B.var));
    for r = 1:length(B.var)
        for s = 1:length(F.var)
            if B.var(r) == F.var(s)
                indB(r) = ind(s);
            end
        end
    end
    Bval = B.val(AssignmentToIndex(indB, B.card));
    F.val(k) = Aval + Bval;
end
end