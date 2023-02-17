% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.  
  MEU = [];
  OptimalDecisionRule = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  singleI = I;
  singleI.UtilityFactors = I.UtilityFactors(1);
  mu = CalculateExpectedUtilityFactor(singleI);
  for i = 2:length(I.UtilityFactors)
      singleI.UtilityFactors = I.UtilityFactors(i);
      newmu = CalculateExpectedUtilityFactor(singleI);
      mu = FactorSum(mu, newmu);
  end
  disp(mu)
  D = I.DecisionFactors(1);
  outputDFactor = struct('var', D.var, 'card', D.card, 'val', zeros(1, prod(D.card)));
  if length(D.card) >= 2
      allEU = zeros(1, D.card(1)^prod(D.card(2:end)));
      NewCard = D.card(1)*ones(1, prod(D.card(2:end)));                    % fits with allEU length
  else
      allEU = zeros(1, D.card(1));
      NewCard = D.card(1);
  end
  END = prod(NewCard);
  Dval = zeros(1, prod(D.card));
  bestDval = repmat([1, 0], 1, prod(D.card(2:end)));
  outputDFactor.val = bestDval;
  besttest = sum(FactorProduct(mu, outputDFactor).val);
  for i = 2:length(allEU)
      ind = IndexToAssignment(i, NewCard);
      for j = 1:length(ind)
          if ind(j) == 1
              Dval(END - 2*j + 1) = 1;
              Dval(END - 2*j + 2) = 0;
          else
              Dval(END - 2*j + 1) = 0;
              Dval(END - 2*j + 2) = 1;
          end
      end
      outputDFactor.val = Dval;
      test = sum(FactorProduct(mu, outputDFactor).val);
      if test > besttest
          bestDval = Dval;
          besttest = test;
      end
  end
  MEU = besttest;
  D.val = bestDval;
  OptimalDecisionRule = CPDFromFactor(D, D.var(1));
  disp(OptimalDecisionRule)
  % A decision rule for D assigns, for each joint assignment to D's parents, 
  % probability 1 to the best option from the EUF for that joint assignment 
  % to D's parents, and 0 otherwise.  Note that when D has no parents, it is
  % a degenerate case we can handle separately for convenience.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  



end
