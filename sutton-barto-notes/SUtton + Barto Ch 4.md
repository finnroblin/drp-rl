

Dynamic Programming

Notes: 
1. Classic DP are not super useful for real-world RL since they assume perfect model and are computationally expensive (NP-hard).
	1. RL algorithms can be thought of as DP algorithms but without as many assumptions/computational power
2. Assumptions:
	1. Environment is finite MDP
	2. We have all state probabilities

Policy Evaluation

1. Consider the Bellman equation
2. Define a series of approximate value functions $v_0, v_1, \dots$
	1. We can approximate the value of some policy $\pi$ as follows:
	2. The sequence $\{v_k\}$ converges to $v_{\pi}$ as $k \to \infty$. 
	3. We can approximate this limit by having some max step size parameter $\theta$, a small value, and repeat until $\Delta < \theta$ for all states $\Delta = max(\Delta, |v-V(s)|$ 
		1. More explicitly, $\Delta = \max_{s \in S} |v_{k+1}(s)-v_k(s)|$ 
3. Producing each next approximation, $v_{k+1}$ from $v_k$, requires us to run: $$v_{k+1}(s) = \sum_a\pi(a|s) \sum_{s',r}p(s',r|s,a)[r+ \gamma v_k(s')]$$

Policy improvement

1. We already have a way to evaluate an entire policy over its action and state space. Now, how do we evaluate different policies and compare different policies?
2. Intuition: Consider if we first take an action $a$ in state $s$ and then follow policy $\pi$ after.
	1. If this is better than following $\pi$ for all states, we should always select $a$ in $s$.
3. The policy improvement theorem follows:
	1. Let $\pi$ and $\pi'$ be any pair of deterministic policies s.t. for all $s \in S$, where $\pi'(s)$ is an action, $$q_{\pi}(s, \pi'(s)) \geq v_{\pi}(s)$$
	2. Then policy $\pi'$ must be as good as or better than $\pi$. $$\forall s \in S, v_{\pi'}(s) \geq v_{\pi}(s)$$
	3. Policy improvement must give us a strictly better policy except when original policy is already optimal.
4. The optimal policy can be determined as follows:
	1. For each state $s$, pick argmax over actions $a$. If this $a$ differs from current policy $\pi(s)$, $\pi$ is not optimal so set $\pi(s) = a$. 
	2. If for all $s$ the policy $\pi$ is optimal, we can conclude that $\pi$ is optimal. 

Value iteration
1. Idea is to sotp policy evaluation after just one step through of the algorithm
2. One sweep of policy evaluation and one sweep of policy improvement

Asynchronous Dynamic Programming 
1. A single sweep over states can be very expensive. 
2. Async DP algos backups states out-of-order. 
3. Can utilize this flexibility to select states faster 
4. Can run an iterative DP algo at the same time an agent experiences the MDP. 

Efficiency of Dynamic programming 