Key ideas:
1. 3 basic ML paradigms: RL, supervised learning, unsupervised learning

Part 1: Introduction, 

Part 2: How to estimate action values with tabular solutions (small state space)
1. DP, Monte Carlo methods, temporal-differences

Part 3: Extending tabular methods to include approximation

Part 4: Frontiers of RL

1. Learning from interactions
2. Basic defintion of RL (full defn = optimal control of MDPs (ch 3))
	1. Basic idea: "Capture the most important aspects of the real problem facing a learning agent interacting with its environment to achieve a goal."
	2. 3 aspects: Agent needs to be able to percieve the state of the environment, and take actions that affect the state. Finaly, the agent must have goal or goals related to state of environment
	3. 3 aspects: Sensation, action, goals
3. Part of trend w/in AI to have greater integration w stats, optimization, other math subjects
	1. Some RL methods can address "curse of dimensioanlity" in OR and control theorty
		1. How can it address this?
4. Some examples
	1. All of these examples involve interaction bw agent and its environment, achieve goal despite uncertainty

Definitons
1. Policy = agent's way of behaving at a given time
2. Reward signal = the goal in a RL problem. At each time step the environment sends a numerical reward to the agent
3. Value function = a value of a state is the expected future reward starting from that state. Values = long-term desirability of state 
4. Model of environment = mimics behavior of environment and allows inferences about how the environment behaves. Methods for solving RL problems that use models and planning are model-based methods, and model-free methods use trial-and-error. Modern ML is across the entire spectrum

Limitations and Scope
1. In this book, estimating value functions is how RL problems are solved, but other solution methods are possible. For example: Genetic algos, genetic programming, simulated annealing, other optimization methods exist without using value functions. 

Tic-tac-toe example