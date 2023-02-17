% Copyright (C) Daphne Koller, Stanford University, 2012

function EU = SimpleCalcExpectedUtility(I)

  % Inputs: An influence diagram, I (as described in the writeup).
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return Value: the expected utility of I
  % Given a fully instantiated influence diagram with a single utility node and decision node,
  % calculate and return the expected utility.  Note - assumes that the decision rule for the 
  % decision node is fully assigned.

  % In this function, we assume there is only one utility node.
  F = [I.RandomFactors I.DecisionFactors];
  U = I.UtilityFactors(1);
  EU = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  EU(1) = 0;
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
  premu = FactorProduct(Fnew, U);
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
  mu = FactorMarginalization(premu, V);
  %disp(mu)
  EU(1) = sum(FactorProduct(mu, D).val);
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
