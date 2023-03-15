
Finite MDPs

Objective: Describe RL problem in a broad sense.

Agent-Environment Interface
1. Agent = Learner and decision maker
2. Environment = everything outside agent that it interacts with
![[drp_agent_env_cycle.png]]
3. For each time step $t$:
	1. Agent recieves state $S_t$
	2. Agent selects an action, $A_t \in A(S_t)$ ($A(S_t)$ = actions available in state $S_t$)
		1. Select action based on your policy
			1. The policy at time $t$ is given by $\pi_t$,
			2. $\pi_t(a|s)$ is probability that $A_t = a$ if $S_t = s$ 
	3. Next, at time $t+1$, agent receives
		1. Numerical reward $R_{t+1}$ and new state $S_{t+1}$
4. Actions are any decision we want to learn how to make, states are any and all useful information
	1. Anything that can't be changed arbitrarily by theagent is considered to be part of the environment (a car's engine is part of the environment, and the agent is the AI doing the driving)

Rewards
1. Maximize the expected return, return $G_t$ is some function of the reward sequence
	1. Episodic tasks have a terminal/ending state. Set of all nonterminal states: $S$, set of all states including terminal state: $S^+$. 
	2. But continuing states, like control problems, don't have a terminal state. So we need discounting for it to converge. 
	3. We write the expected discounted return $G_t$ as: $$G_t = R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma ^ k R_{t+k+1}$$
	4. With $0 \leq \gamma \leq 1$, and higher $\gamma$ = considers farther away rewards more valuable. $\lambda = 0$ implies that agent maximizes only the next reward 
	5. Generalizing between episodic and continunuing tasks requires another formula: $$G_t = R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}$$
Markov Property
1. A Markov property exists when 
	1. State signal is given to the agent by the environment, but does not reveal everything about the environment and trivialize decisions
		1. An agent playing chess gets the entire board as a signal, but an agent playing poker doesn't get to see the other players' hands (but can perhaps see their facial expressions)
2. Markov property idea:
	1. We can model the dynamics at time $t+1$ for all possible rewards and state transitions as: $$Pr\{{R_{t+1} = r, S_{t+1} = s' | S_0, A_0, R_1, \dots, S_{t-1}, A_{t-1}, R_t, S_t, A_t}\}$$
	2. If a RL problem has the Markov property, it is said to be memoryless, meaning we can define the complete dynamics just as well with: $$p(s', r|s,a) = Pr\{{R_{t+1} = r, S_{t+1} = s' | S_t, A_t}\}$$
	3. Normally it's okay to think of state at each time step as an approximation to a Markov state, even if it doesn't fully satisfy Markov property
		1. Think about a very good poker player — they're not keeping in mind all previous states, but still perform just about as well as if they could. 

Markov Decision Process
1. RL task satisfying Markov property is a Markov decision process, or MDP. Finite MDPs have finite state and action spaces, account for 90% of RL
	1. All dynamics rest on: $$p(s', r|s,a) = Pr\{{R_{t+1} = r, S_{t+1} = s' | S_t, A_t}\}$$
	2. Expected rewards for state-action pairs: $$r(s,a) = \mathbb{E}[R_{t+1} | S_t =s, A_t=a]=\sum_{r\in R}{r}\sum_{s' \in S}p(s', r|s,a)$$
	3. State transition probabilities: $$p(s'|s,a) = Pr\{{ S_{t+1} = s' | S_t=s, A_t=a}\}$$
	4. Expected rewards for state, action, next-state triples: $$r(s,a,s') = \mathbb{E}[R_{t+1}|S_t=s,A_t=a,S_{t+1} = s']=\frac{\sum_{r \in R}rp(s',r|s,a)}{p(s'|s,a)}$$

Value Functions
1. Estimating value functions is an important part of RL algorithms — they estimate how good it is for an agent to perform an action in a given state. "How good" is defined in terms of future expected return (future expected rewards).
	1. Thus value functions are defined wrt to policies, $\pi$.
2. $v_{\pi}(s)$ is the state-value function for policy $\pi$ — it is the expected return when starting in $s$ and following $\pi$ after. $$v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s] = \mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s]$$
3. $q_{\pi}(s,a)$ is the action-value function for policy $\pi$ — it is the expected return of starting in state $s$, taking the action $a$, and thereafter following policy $\pi$. $$q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t=a] = \mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s,A_t=a]$$
4. Let us now expand $v_{\pi}(s)$: $$v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s] = \mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s]$$
	1. Setting $k = k+1$, and then applying linearity of expectations $$=\mathbb{E}_{\pi}[R_{t+1}+\gamma \sum_{k=0}^{\infty}\gamma^kR_{t+k+2}|S_t=s]=\mathbb{E}_{\pi}[R_{t+1}]+\mathbb{E}_{\pi}[\gamma \sum_{k=0}^{\infty}\gamma^kR_{t+k+2}|S_t=s]$$
		1. Simplifying $\mathbb{E}_{\pi}[R_{t+1}]$, where $\pi(a|s)$ is probability of taking action $a$ in state $s$  $$\mathbb{E}_{\pi}[R_{t+1}] = \sum_{a}\pi(a|s)\mathbb{E}[R_{t+1} | S_t =s, A_t=a]=\sum_{a}\pi(a|s)\sum_{r\in R}{r}\sum_{s' \in S}p(s', r|s,a)$$
		2. Simplifying $E_{\pi}[\gamma \sum_{k=0}^{\infty}\gamma^kR_{t+k+2}|S_t=s]$ 
			1. First, let $G_{t+1} = \sum_{k=0}^{\infty}\gamma^kR_{t+k+2}$ . $G_{t+1}$ is a RV taking on finite values $g \in \Gamma$. Then $$\mathbb{E}_{\pi}[G_{t+1}|S_t=s]=\sum_{g \in \Gamma}gp(g|s)$$
			2. We now manipulate $p(g|s)$ 
				1. First rewrite it to depend on $s',r,a$: $$p(g|s)=\sum_{r} \sum_{s'} \sum_a p(s',r,a,g|s)$$
				2. Then use the law of multiplication for joint probabilties, (remembering that $p(a|s) = \pi(a|s)$ yielding: $$=p(g|s)=\sum_{r} \sum_{s'} \sum_a p(g|s',r,a,s)p(s',r|a,s)\pi (a|s)$$
				3. Finally, we use the fact that $g$ depends only on $s'$, and not $r,a,s$ since we already know $s'$, and since the process is Markovian, it does not depend on $r,a,s$. 
					1. Thus $p(g|s',r,a,s)=p(g|s')$, and we have: $$p(g|s)=\sum_{r} \sum_{s'} \sum_a p(g|s')p(s',r|a,s)\pi (a|s)$$
			3. Now, using expanded $p(g|s)$: $$\sum_{g \in \Gamma} gp(g|s) = \sum_{r} \sum_{s'} \sum_a \sum_{g\in \Gamma} p(g|s')p(s',r|a,s)\pi (a|s)$$
			4. Using the fact that by our definition of $=\mathbb{E}_{\pi}[G_{t+1}|S_{t+1}=s']$, we have: $\mathbb{E}[G_{t+1}|s']=\sum_{g\in \Gamma} gp(g|s')$  ,  $$=\sum_{r} \sum_{s'} \sum_a  \mathbb{E}_{\pi}[G_{t+1}|S_{t+1}=s']p(s',r|a,s)\pi (a|s)$$
			5. Finally, using $\mathbb{E}_{\pi}[G_{t+1}|S_{t+1}=s']=v_{\pi}(s')$, we have $$=\sum_{r} \sum_{s'} \sum_a  v_{\pi}(s')p(s',r|a,s)\pi (a|s)$$
		3. Finally, $$\gamma \mathbb{E}_{\pi}[ \sum_{k=0}^{\infty}\gamma^kR_{t+k+2}|S_t=s]=\gamma\sum_{r} \sum_{s'} \sum_a  v_{\pi}(s')p(s',r|a,s)\pi (a|s)$$
	2. We can combine the two expectations: $$\sum_{a}\pi(a|s)\sum_{r\in R}{r}\sum_{s' \in S}p(s', r|s,a) + \gamma\sum_{r} \sum_{s'} \sum_a  v_{\pi}(s')p(s',r|a,s)\pi (a|s)$$ $$v_{\pi}(s)=  \sum_a \pi (a|s)\sum_{r} \sum_{s'} v_{\pi}(s')p(s',r|a,s)[r+ \gamma v_{\pi}(s')]$$
5. Finally, we have the Bellman equation: $$v_{\pi}(s)=  \sum_a \pi (a|s)\sum_{r} \sum_{s'} p(s',r|a,s)[r+ \gamma v_{\pi}(s')]$$
 6. The Bellman equation is recursive and allows us to compute the value of a state from its successor state $S_{t+1} = s'$ 

Optimal value functions:
1. Solving a RL task = find the **best** policy. For finite MDPs:
	1. Value functions define a partial ordering over policies
		1. $\pi \geq \pi'$ iff $v_{\pi}(s) \geq v_{\pi'}(s) \forall s \in S$ 
	2. All optimal policies are denoted $\pi_*$ 
		1. $$v_*(s)=\max_{\pi} v_{\pi}(s)$$  for all $s$
	3. The optimal action-value function, $q_*$ is: $$q_*(s,a)=\max_{\pi} q_{\pi}(s,a)$$
	4. We can also write $q_*$ in terms of $v_*$ since $q_*$ gives us the expected return for action $a$ in state $s$ and following an optimal policy afterwards $$q_*(s,a)=\mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1}) | S_t=s, A_t=a]$$
2. Since $v_*$ is the optimal value function, we don't need to reference any specific policy. The maximum return is under an optimal policy for that state. Thus we have the Bellman optimality equations:
	1. In terms of expectation: $$v_*(s) = \max_a \mathbb{E} [R_{t+1}+\gamma v_*(S_{t+1})|S_t=s,A_t=a]$$
	2. An in terms of probailities: $$\max_{a \in A(s)} \sum_{s',r}p(s',r|s,a)[r+\gamma v_*(s')]$$
		1. This equation has a unique solution independent of the policy — it's a system of equations for N states, N equations and in N unkowns. If $p(s',r|s,a)$ is known then the system of equations can be solved. 
	3. Bellman optimality for $q_*$: $$q_*(s,a)=  \mathbb{E} [R_{t+1}+\gamma \max_{a'} q_*(S_{t+1},a')|S_t=s,A_t=a]$$ and $$q_*(s,a)= \sum_{s',r}p(s',r|s,a)[r+\gamma \max_{a'} q_*(s',a')]$$
3. When we have solved $v_*$, we can easily determine and optimal policy. To determine an optimal policy we assign nonzero probabilities only to the one or more actions per step that lead to a maximum being obtained for each $q$. 
	1. This greedy policy is actually optimal, since $v_*$ takes into account all long-term rewards through its formulation. 

Approximation algorithms
1. E.g. for Backgammon, there are $10^{20}$ states so finding optimal solution is not possible.
2. Tabular methods are available for small state sets where we form approximations in tables with one entry per state/state-action pair. 