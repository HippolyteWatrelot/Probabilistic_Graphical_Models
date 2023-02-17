
function Fnew = Reorder(F)
Fnew = struct('var', [], 'card', [], 'val', zeros(1, prod(F.card)));
Fnew.var = sort(F.var);
for i = 1:length(Fnew.var)
    ind = find(F.var == Fnew.var(i));
    Fnew.card(i) = F.card(ind);
end
for k = 1:prod(F.card)
    F_ass = IndexToAssignment(k, F.card);
    Fnew_ass = zeros(1, length(F_ass));
    for r = 1:length(F_ass)
        ind = Fnew.var == F.var(r);
        Fnew_ass(ind) = F_ass(r);
    end
    Fnew_index = AssignmentToIndex(Fnew_ass, Fnew.card);
    Fnew.val(Fnew_index) = F.val(k);
end
end