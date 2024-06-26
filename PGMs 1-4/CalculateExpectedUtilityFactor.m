% Copyright (C) Daphne Koller, Stanford University, 2012

function EUF = CalculateExpectedUtilityFactor( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: A factor over the scope of the decision rule D from I that
  % gives the conditional utility given each assignment for D.var
  %
  % Note - We assume I has a single decision node and utility node.
  EUF = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  D = I.DecisionFactors;
  if length(I.RandomFactors) >= 2
      Fnew = I.RandomFactors(1);
      for i = 2:length(I.RandomFactors)
          Fnew = FactorProduct(Fnew, I.RandomFactors(i));
      end
  else
      Fnew = I.RandomFactors;
  end
  %disp(Fnew)
  premu = FactorProduct(Fnew, I.UtilityFactors);
  %disp(premu)
  V = [];
  index = 1;
  for j = 1:length(premu.var)
      if ismember(premu.var(j), D.var) == 0
          V(index) = premu.var(j);
          index = index + 1;
      end
  end
  %disp(V)
  EUF = FactorMarginalization(premu, V);           % Which is mu
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

