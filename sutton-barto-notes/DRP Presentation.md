
1. Explain motivations for reinforcement learning
2. Ch 1: Introduce RL 
	1. Why RL is useful
	2. Background of RL — place it in context of broader ML trends 
		1. RL closer to mathematical optimization, how humans and animals learn, 
	3. Bring in modern examples of playing starcraft or some other novel applications (that will be readdressed at the end)
	4. Define the concept of an agent and value function
3. Ch 2: Motivate modern RL with multi-arm bandits
	1. Use multi-arm bandits to introduce policies
	2. Talk about greedy, e-greedy, optimistic, etc
	3. Bring in generated graphs, discuss tradeoffs of each policy
4. Ch 3: Talk about finite MDPs
	1. Transition to MDPs from bandits by explaining the more expansive environment
	2. MDPs are like bandits, but with the rewards being state-dependent 
	3. Define stochastic policy and show some diagrams
	4. Markov property
	5. => MDPs (that satisfy Markov property)
	6. "Finite MDPs are all you need to understand 90% of modern reinforcement learning"
	7. Bellman equations
	8. Optimal value functions
5. Ch 4: Dynamic programming to solve MDPs
	1. Gridworld
	2. Policy evaluation — idea of backups and all rewards being the consequence of past rewards
	3. Greedy policy improvement
	4. Fact that policy converges in surprisginly few iteratiions
	5. Async DP & limitations of DP
6. Other applications
	1. Deep Q-learning at high level
