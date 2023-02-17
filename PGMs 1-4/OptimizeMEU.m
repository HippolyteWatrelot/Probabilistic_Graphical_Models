% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  %EUF = CalculateExpectedUtilityFactor(I);
  if length(D.card) >= 2
      allEU = zeros(1, D.card(1)^prod(D.card(2:end)));
      NewCard = D.card(1)*ones(1, prod(D.card(2:end)));                    % fits with allEU length
  else
      allEU = zeros(1, D.card(1));
      NewCard = D.card(1);
  end
  Dval = zeros(1, prod(D.card));
  %disp(Dval)
  END = length(Dval);
  for k = 1:length(allEU)
      K = IndexToAssignment(k, NewCard);
      for j = 1:length(K)
          if K(j) == 1
              Dval(END - 2*j + 1) = 1;
              Dval(END - 2*j + 2) = 0;
          else
              Dval(END - 2*j + 1) = 0;
              Dval(END - 2*j + 2) = 1;
          end
      end
      %disp(Dval)
      I.DecisionFactors(1).val = Dval;
      allEU(k) = SimpleCalcExpectedUtility(I);
  end
  disp(allEU)
  [MEU, index]= max(allEU);
  IND = IndexToAssignment(index, NewCard);
  bestDval = zeros(1, length(D.val));
  for i = 1:length(IND)
      if IND(i) == 1
          bestDval(END - 2*i + 1) = 1;
          bestDval(END - 2*i + 2) = 0;
      else
          bestDval(END - 2*i + 1) = 0;
          bestDval(END - 2*i + 2) = 1;
      end
  end
  OptimalDecisionRule = struct('var', D.var, 'card', D.card, 'val', bestDval);
  OptimalDecisionRule = CPDFromFactor(OptimalDecisionRule, D.var(1));
  if length(OptimalDecisionRule.var) == 1
      OptimalDecisionRule.var(2) = 0;
  end
  PrintFactor(OptimalDecisionRule)
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    

end
