
Multiarm bandits

Differentiator b/w RL and other learning: 
It uses training information that evaluates the actions taken, rather than instructing by giving correct actions. This is a paradigm that's distinct from other machine learning. 

Bandits are nonassociative, meaning there is only one state and one situation

The idea behidn n-armed bandit problem comes from slot machines in casinos = "bandits." Bandits have some number of arms, each of which can also be called an action value. You want to pull the lever that has the biggest expected reward and maximize total reward.
If you take a greedy action — pulling the lever whose EV is greatest — you are exploiting your current knowledge of the action values. If you take a nongreedy action — pulling a lever randomly — you are exploring the space since it allows you to improve your estimate of the nongreedy action's value. Exploration might maximize total reward in the long run. The tradeoff is very compliacted, so we worry only about balancing them at all.

Define an action (lever) $a$. $q(a)$ is the actual value of the lever/action. Let $Q_t(a)$ be the estimated value of action $a$ after $N_t(a)$ pulls at timestep $t$. Then $R_{N_t(a)}$ is the reward at the Nth pull of action a at time t. 
$$Q_t(a) = \frac{R_1 + R_2 + \dots + R_{N_t(a)}}{N_t(a)}$$

We want to take $a$ to maximize total reward

Let $Q_k(a) = Q_k$ be the estimate of action $a$'s $k$th reward.
$$Q_{k+1} = Q_k + \frac{1}{k}[R_k - Q_k]$$ Where $\frac{1}{k}$ is the step-size parameter. This look similar to gradient decent. 

If the bandit changes over time, we want to prioritize more recentl estimates since they're likely to be better. The higher $\alpha$, the more priority recent rewards are given and the more past rewards are discounted. 