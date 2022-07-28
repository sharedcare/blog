---
layout: pseudo
title:  "DQN is All You Need"
date:   2022-05-07 00:00:00 +0800
categories: intro
mathjax: True
mathjax_autoNumber: True
---
# DQN is All You Need

Rainbow is a method to combine all six improvements in Deep Q-Network(DQN). Since the emergence of DQN, scholars have proposed various improvements to it. This post will focus on six of the major improvements, which are eventually integrated into the **Rainbow** model.

# Deep Q-Network

As I introduced in my previous post, the **Dynamic Programming**, **Monte Carlo**, and **Temporal Difference** approach only work on discrete finite sets of states. For complicated and continuous states, we need to use neural networks to approximate the state value function.

Unlike the basic **DQN**, we adopt the improved version of **DQN**(Mnih, Volodymyr, et al.) which uses two neural networks in practice. In the ordinary **DQN**, we use a Q-network as the policy network. Whereas in the improved **DQN**, in addition to the policy network, we also need a target network to calculate the target Q value.

{% include pseudocode.html id="1" code="
\begin{algorithm}
\caption{Deep Q-learning with Experience Replay}
\begin{algorithmic}
\STATE \textbf{Algorithm parameter:} memory size $N$, step size $\alpha \in (0, 1]$, small $\epsilon > 0$
\STATE \textbf{Initialize:} 
    \STATE Replay memory $\mathcal{D}$ to capacity $N$
    \STATE Action-value function $Q$ with random weights $\theta$
    \STATE Target action-value function $\hat{Q}$ with weights $\theta^-=\theta$
\For{each episode($episode = 1,..., M$)}
    \STATE \textbf{Initialize} $s_1$ and \textbf{preprocessed} sequenced feature vector $\phi_1 = \phi(s_1)$
    \For{each step of episode($t = 1,..., T$)}
        \REPEAT
        \STATE Choose $a_t$ from $s_t$ using $\epsilon$-greedy policy derived from $Q$ where $a_t = \max_a Q^*(\phi(s_t), a; \theta)$
        \STATE Execute action $a_t$, observe $r_t$, $s_{t+1}$, and terminal state $is\_end_t$
        \STATE Set $s_{t+1}= s_t, a_t$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
        \STATE Store transition $(\phi_t, a_t, r_t, \phi_{t+1}, is\_end_t)$ in $\mathcal{D}$
        \STATE Sample random mini-batch of transitions $(\phi_j, a_j, r_j, \phi_{j+1}, is\_end_j)$ from $\mathcal{D}$
        \STATE Set $y_j= \begin{cases} r_j& {\text{if }is\_end_j\text{ is true}}\\ r_j + \gamma\max_{a'}Q(\phi_{j+1},a';\theta) & {\text{if }is\_end_j\text{ is false}} \end{cases}$
        \STATE Perform a gradient descent step using $\frac{1}{m}\sum\limits_{j=1}^m(y_j-Q(\phi_j,a_j;\theta))^2$ to update the parameters $\theta$ in Q-Network
        \STATE every $C$ steps reset $\hat{Q}=Q$
        \UNTIL{$s_t$ is terminal}
    \EndFor
\EndFor
\end{algorithmic}
\end{algorithm}
" %}

# Double DQN

**Double DQN(DDQN)** separates action selections and value estimation to avoid overestimation of the value, since **DQN** uses a greedy policy to estimate the target Q value, this case will always overestimate the Q value. **DDQN** is basically similar to the above **DQN**, the only difference is the method of calculating the target Q value.

Instead of finding the maximum Q value in each action with the target Q network, **DDQN** first finds the optimal action corresponding to the maximum Q value in the current policy Q network:

$$
a^{\max}(S'_j, \theta) = \arg\max_{a'}Q(\phi(S'_j),a,\theta)
$$

Then use this selected action $a^{\max}(S_j',\theta)$ to calculate the target Q value in the target Q network:

$$
y_j = r_j + \gamma Q'(\phi(S'_j),\arg\max_{a'}Q(\phi(S'_j),a,w),w')
$$

{% include pseudocode.html id="2" code="
\begin{algorithm}
\caption{Double Deep Q-learning}
\begin{algorithmic}
\STATE \textbf{Algorithm parameter:} memory size $N$, step size $\alpha \in (0, 1]$, small $\epsilon > 0$
\STATE \textbf{Initialize:} 
    \STATE Replay memory $\mathcal{D}$ to capacity $N$
    \STATE Action-value function $Q$ with random weights $\theta$
    \STATE Target action-value function $\hat{Q}$ with weights $\theta^-=\theta$
\For{each episode($episode = 1,..., M$)}
    \STATE \textbf{Initialize} $s_1$ and \textbf{preprocessed} sequenced feature vector $\phi_1 = \phi(s_1)$
    \For{each step of episode($t = 1,..., T$)}
        \REPEAT
        \STATE Choose $a_t$ from $s_t$ using $\epsilon$-greedy policy derived from $Q$ where $a_t = \max_a Q^*(\phi(s_t), a; \theta)$
        \STATE Execute action $a_t$, observe $r_t$, $s_{t+1}$, and terminal state $is\_end_t$
        \STATE Set $s_{t+1}= s_t, a_t$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
        \STATE Store transition $(\phi_t, a_t, r_t, \phi_{t+1}, is\_end_t)$ in $\mathcal{D}$
        \STATE Sample random mini-batch of transitions $(\phi_j, a_j, r_j, \phi_{j+1}, is\_end_j)$ from $\mathcal{D}$
        \STATE Set $y_j= \begin{cases} r_j& {\text{if }is\_end_j\text{ is true}}\\ r_j + \gamma Q'(\phi_{j+1},\arg\max_{a'}Q(\phi_{j+1},a;\theta);\theta') & {\text{if }is\_end_j\text{ is false}} \end{cases}$
        \STATE Perform a gradient descent step using $\frac{1}{m}\sum\limits_{j=1}^m(y_j-Q(\phi_j,a_j;\theta))^2$ to update the parameters $\theta$ in Q-Network
        \STATE every $C$ steps reset $\hat{Q}=Q$
        \UNTIL{$s_t$ is terminal}
    \EndFor
\EndFor
\end{algorithmic}
\end{algorithm}
" %}

# Dueling Networks

**Dueling DQN** decomposes Q value into state value and advantage function to get more useful information.

In order to optimize the network structure, **Dueling DQN** considers dividing the Q-network into two parts: the first part is only related to the state $s$ and not to the specific action $a$ to be executed, this part is called the value function, denoted as $V(s;\theta,\alpha)$; the second part is related to both the state $s$ and the action $a$, which is called the Advantage Function, denoted as $A(s,a;\theta,\beta)$. Then our value function can be defined as:

<div class="card" style="margin: auto; max-width: fit-content;" markdown="0">
  <div class="card__image">
    <img class="image" src="/assets/imgs/2022-05-07-rainbow-dqn/1.png"/>
  </div>
  <div class="card__content">
    <p>Regular deep Q-network structure (<strong>top</strong>) and the dueling Q-network (<strong>bottom</strong>) [Source: Wang et al., [Dueling Network Architectures for Deep Reinforcement Learning]</p>
  </div>
</div>

$$
Q(s,a;\theta, \alpha, \beta) = V(s;\theta,\alpha) + A(s,a;\theta,\beta)
$$

where $\theta$ is the shared parameter, $\alpha$ is the parameter of the value function, and $\beta$ is the parameter of the advantage function.

To distinguish the value of $V$ and $A$ in $Q$, the actual formulation chosen for identifiability is defined as:

$$
 Q\left(s, a; \theta, \alpha, \beta\right) =V\left(s; \theta, \beta\right) + \left(A\left(s, a; \theta, \alpha\right) - \frac{1}{|\mathcal{A}|}\sum_{a'}A\left(s, a'; \theta, \alpha\right)\right) 
$$

In this case, the advantage function estimator has zero advantage at the chosen action. Instead of calculating the maximum, it uses an average operator to increase the stability of the optimization.

# Prioritized Experience Replay

All types of **DQN** are sampled by experience replay, and then do the calculation of the target Q value. In general, **DQN** collects experiences by random sampling, so all samples have the same probability of being sampled.

However, different samples in the **Experience Replay** pool have different **TD errors $\vert\delta_t\vert$** , in this case, they have different effects on the backpropagation process. In Q-networks, the **TD error** represents the difference between the target Q value(value of TD target) and the current Q value(estimation of Q). The larger the **TD error**, the greater the effect on the backpropagation. While with a small **TD error**, the samples will have little effect on the backpropagation. If samples with larger **TD error** are more likely to be sampled, it is easier to converge the deep Q-learning algorithm.

## Prioritizing with TD-error

In addition to the **Experience Replay** in the regular **DQN**, the **Prioritized Experience Replay(PER)** not only stores state, action, reward, and other data, and also add priority to making the order for sampling. In this case, we use **TD-error** to indicate the priority of each transition. One approach is to use a greedy **TD-error** prioritization to rank the new transitions with the unknown **TD-error** in the highest priority. However, this can also have some issues at the same time:

1. samples with low **TD-error** may never be replayed.
2. sensitive to noise.
3. samples with large **TD-error** can easily make the neural network overfit (because transitions with large **TD-error** can be replayed frequently).

### Stochastic Prioritization

To solved the above problems, **Stochastic Prioritization** is introduced. Concretely, define the probability of sampling transition $i$ as:

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}
$$

where $p_i$ is the priority of transition $i$. $\alpha$ indicates how much prioritization is used. $\alpha=0$ is equivalent to the general uniform sampling.

For the above $P(i)$, there are two variants:

1. **Proportional prioritization:** 
    
    Define $p_i=\vert\delta_i\vert+\epsilon$, where $\delta_i$ is the TD-error and $\epsilon$ denotes a very small positive number. The purpose of this is to make sure that the transitions with a TD-error of 0 will also be sampled. In practice, we use a ‘sum-tree’ data structure as the experience buffer pool for this variant**.**
    
2. **Rank-based prioritization:** 
    
    Define $p_i = \frac{1}{\text{rank}(i)}$, where $\text{rank}(i)$ is the ranking according to $|\delta_i|$. In this case, $P$
     becomes a power-law distribution with exponent $\alpha$.
    

However, the **PER** also introduces bias since the distribution of the samples is changed. This may cause our model to converge to a different value. Thus, we introduce **importance sampling(IS)** to correct the bias:

$$
 w_{i} = \left(\frac{1}{N}\cdot\frac{1}{P\left(i\right)}\right)^{\beta} 
$$

{% include pseudocode.html id="3" code="
\begin{algorithm}
\caption{Double DQN with proportional prioritization}
\begin{algorithmic}
\STATE \textbf{Algorithm parameter:} minibatch $k$, replay period $K$ memory size $N$, step size $\eta \in (0, 1]$, exponents $\alpha$ and $\beta$, budget $T$.
\STATE \textbf{Initialize:} 
    \STATE Replay memory $\mathcal{D} = \emptyset,\Delta=0,p_1=1$
    \STATE Action-value function $Q$ with random weights $\theta$
    \STATE Target action-value function $\hat{Q}$ with weights $\theta^-=\theta$
\For{each episode($episode = 1,..., M$)}
    \STATE \textbf{Initialize} $s_1$ and \textbf{preprocessed} sequenced feature vector $\phi_1 = \phi(s_1)$
    \For{each step of episode($t = 1,..., T$)}
        \REPEAT
        \STATE Choose $a_t$ from $s_t$ using $\epsilon$-greedy policy derived from $Q$ where $a_t = \max_a Q^*(\phi(s_t), a; \theta)$
        \STATE Execute action $a_t$, observe $r_t$, $s_{t+1}$, and terminal state $is\_end_t$
        \STATE Set $s_{t+1}= s_t, a_t$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
        \STATE Store transition $(\phi_t, a_t, r_t, \phi_{t+1}, is\_end_t)$ in $\mathcal{D}$ with maximal priority $p_t = \max_{i<t} p_i$
        \IF{$t\equiv 0\mod K$}
        \FOR{$j = 1$ to $k$}
        \STATE Sample transitions $(\phi_j, a_j, r_j, \phi_{j+1}, is\_end_j)$ from $\mathcal{D}$ with $P(j) = \frac{p_j^\alpha}{\sum\limits_i(p_i^\alpha)}$
        \STATE Compute importance-sampling weight $w_j = (N\cdot P(j))^{-\beta}/\max_i(w_i)$
        \STATE Set $y_j= \begin{cases} r_j& {\text{if }is\_end_j\text{ is true}}\\ r_j + \gamma Q'(\phi_{j+1},\arg\max_{a'}Q(\phi_{j+1},a;\theta);\theta') & {\text{if }is\_end_j\text{ is false}} \end{cases}$
        \STATE Compute TD-error $\delta_j = y_j- Q(\phi_j,a_j;\theta)$
        \STATE Update transition priority $p_j \leftarrow |\delta_j |$
        \STATE Accumulate weight-change $\Delta \leftarrow \Delta + w_j · \delta_j · \nabla_\theta Q(s_{j}, a_{j})$
        \ENDFOR
        \STATE Update weights $\theta \leftarrow \theta + \eta \cdot \Delta$, reset $\Delta = 0$
        \STATE every $C$ steps reset $\hat{Q}=Q$
        \ENDIF
        \UNTIL{$s_t$ is terminal}
    \EndFor
\EndFor
\end{algorithmic}
\end{algorithm}
" %}

# Multi-Step Learning

Similar to the **TD($\lambda$)** method instead of using a single step approach in regular DQN, Multi-step learning considers $n$-step of the rewards:

$$
R_t^{(n)}=\sum_{k=0}^{n-1}\gamma_t^{(k)}R_{t+k+1}
$$

By estimating the target value through the rewards of $n$ steps, the loss function of the **DQN** becomes:

$$
(R_t^{(n)}+\gamma_t^{(n)}\max_{a'}q_{\theta'}(S_{t+n},a')-q_\theta(S_t,A_t))^2
$$

This approach can help us speed up the training of **DQN** agents.

# Distributional RL

### Bellman Operator

In previous post, we define Bellman equations as:

$$
v_{\pi}(s) = \sum\limits_{a \in A} \pi(a\vert s)(r(s,a) + \gamma \sum\limits_{s' \in S}P(s'\vert s,a)v_{\pi}(s'))
$$

$$
q_{\pi}(s,a) = r(s, a) + \gamma \sum\limits_{s' \in S}P(s'\vert s,a)\sum\limits_{a' \in A} \pi(a'\vert s')q_{\pi}(s',a')
$$

When $\pi$ is deterministic, the equations can be reduced to:

$$
v_{\pi}(s) = r(s,\pi(s)) + \gamma \sum\limits_{s' \in S}P(s'\vert s,a)v_{\pi}(s')
$$

$$
q_{\pi}(s,a) = r(s, a) + \gamma \sum\limits_{s' \in S}P(s'\vert s,a)q_{\pi}(s',\pi(s'))
$$

Define the Bellman operator $T^\pi : (S \rightarrow \mathbb{R}) \rightarrow (S \rightarrow \mathbb{R})$ for $S$ via any $v : S \rightarrow \mathbb{R}$ in the following way:

$$
(T^\pi v)(s) = r(s,\pi(s)) + \gamma \sum\limits_{s' \in S}P(s'\vert s,a)v_{\pi}(s')
$$

Similarly, define the Bellman operator for functions of $S \times A$ as $T^\pi : (S \times A \rightarrow \mathbb{R}) \rightarrow (S \times A \rightarrow \mathbb{R})$:

$$
(T^\pi q)(s,a) = r(s, a) + \gamma \sum\limits_{s' \in S}P(s'\vert s,a)q_{\pi}(s',\pi(s'))
$$

Thus, we can rewrite the Bellman equations as:

$$
T^\pi v_\pi=v_\pi,T^\pi Q_\pi=Q_\pi
$$

Then, the Bellman optimality operators can be defined as:

$$
(T v)(s) = \max_a (r(s,a) + \gamma \sum\limits_{s' \in S}P(s'\vert s,a)v(s'))
$$

$$
(Tq)(s,a) = r(s, a) + \gamma \sum\limits_{s' \in S}P(s'\vert s,a)\max_{a'\in A}q(s',a')
$$

The Bellman operator $\mathcal{T}^{\pi}$ is a $\gamma$*-contraction*, meaning that for any $q_1,q_2$ we have:

$$
\mathrm{dist}\left(\mathcal{T}Q_{1},\mathcal{T}Q_{2}\right)\leq\gamma\mathrm{dist}\left(Q_{1},Q_{2}\right)
$$

Thus according to the *Contraction Mapping Theorem*, $\mathcal{T}^{\pi}$ has a unique fixed point, and then we have:

$$
\mathcal{T}^{\infty}Q=Q^{\pi}
$$

### KL Divergence

For the distribution $p,q$, the KL Divergence is defined as:

$$
\mathrm{KL}(p\|q)=\int p(x)\log\frac{p(x)}{q(x)}dx
$$

As for the **discrete** case:

$$
\mathrm{KL}(p\|q)=\sum_{i=1}^{N}p(x_{i})\log\frac{p(x_{i})}{q(x_{i})}=\sum_{i=1}^{N}p(x_{i})[\log p(x_{i})-\log q(x_{i})]
$$

---

Distributional DQN replaces the value function in a regular DQN with a value distribution. Therefore, the value function $q(s,a)$ becomes a value distribution in Distributional DQN. It receives $s,a$ and outputs a distribution that describes all values of the state action pair $(s,a)$.

<div class="card" style="margin: auto; max-width: fit-content;" markdown="0">
  <div class="card__image">
    <img class="image" src="/assets/imgs/2022-05-07-rainbow-dqn/2.png"/>
  </div>
</div>

In Distributional DQN, we no longer use $Q(s,a)$ for "value", we no longer use the expectation to estimate the value, but use its distribution directly, $Q(s,a)$ and $Z(x,a)$ satisfy

$$
Q(s,a)=\mathbb{E}[Z(s,a)]=\sum_{i=1}^{N}p_{i}x_{i}
$$

Our purpose is to move $Z(s,a)$ towards $r+\gamma Z(s',a^{*})$, where $r+\gamma Z(s',a^{\ast})$ is a collection of samples of the real target distribution.

{% include pseudocode.html id="4" code="
\begin{algorithm}
\caption{Deep Q-learning with Categorical Algorithm}
\begin{algorithmic}
\STATE \textbf{Algorithm parameter:} memory size $N$, step size $\alpha \in (0, 1]$, small $\epsilon > 0$
\STATE \textbf{Initialize:} 
    \STATE Replay memory $\mathcal{D}$ to capacity $N$
    \STATE Action-value function $Q$ with random weights $\theta$
    \STATE Target action-value function $\hat{Q}$ with weights $\theta^-=\theta$
\For{each episode($episode = 1,..., M$)}
    \STATE \textbf{Initialize} $s_1$ and \textbf{preprocessed} sequenced feature vector $\phi_1 = \phi(s_1)$
    \For{each step of episode($t = 1,..., T$)}
        \REPEAT
        \STATE Choose $a_t$ from $s_t$ using $\epsilon$-greedy policy derived from $Q$ where $a_t = \max_a Q^*(\phi(s_t), a; \theta)$
        \STATE Execute action $a_t$, observe $r_t$, $s_{t+1}$, and terminal state $is\_end_t$
        \STATE Set $s_{t+1}= s_t, a_t$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
        \STATE Store transition $(\phi_t, a_t, r_t, \phi_{t+1}, is\_end_t)$ in $\mathcal{D}$
        \STATE Sample random mini-batch of transitions $(\phi_j, a_j, r_j, \phi_{j+1}, is\_end_j)$ from $\mathcal{D}$
        \STATE $Q(\phi_{j+1},a)=\sum_i z_ip_i(\phi_{j+1},a)$
        \STATE $a_*\leftarrow \arg\max_aQ(\phi_{j+1},a)$
        \STATE $m_i=0,i\in0,...,N-1$
        \FOR{$k\in0,...,N-1$}
        \STATE $\hat{\mathcal{T}}z_k\leftarrow[r_j+\gamma_jz_k]^{V_{\max}}_{V_{\min}}$ \COMMENT{Computer the projection of $\hat{\mathcal{T}}z_k$ onto the support $\{z_i\}$}
        \STATE $b_k\leftarrow(\hat{\mathcal{T}}z_k-V_{\min})/\Delta z$   \COMMENT{$b_k\in[0,N-1]$}
        \STATE $l\leftarrow \lfloor b_k\rfloor,u\leftarrow \lceil b_k\rceil$
        \STATE $m_l\leftarrow m_l+p_k(\phi_{j+1},a_*)(u-b_k)$ \COMMENT{Distribute probability of $\hat{\mathcal{T}}z_k$}
        \STATE $m_u\leftarrow m_u+p_k(\phi_{j+1},a_*)(b_k-l)$
        \ENDFOR
        \STATE $loss = -\sum_im_i\log p_i(\phi_j,a_j)$   \COMMENT{Cross-entropy loss}
        \STATE Perform a gradient descent step using $\frac{1}{m}\sum\limits_{j=1}^m(y_j-Q(\phi_j,a_j;\theta))^2$ to update the parameters $\theta$ in Q-Network
        \STATE every $C$ steps reset $\hat{Q}=Q$
        \UNTIL{$s_t$ is terminal}
    \EndFor
\EndFor
\end{algorithmic}
\end{algorithm}
" %}

# Noisy Nets

The exploration capability of the Agent can be effectively enhanced by using **Noisy Net**. This approach increases the exploration capability of the model by adding noise to the parameters. In this case, the linear layer in the model becomes a noisy linear layer:

$$
y=(b+Wx)+(b_{noisy}\odot\epsilon^b+(W_{noisy}\odot\epsilon^w)x)
$$

where $\epsilon$ is a random noise which is following the standard normal distribution $N(0,1)$.

# References

1. Hessel, Matteo, et al. "Rainbow: Combining improvements in deep reinforcement learning." *Thirty-second AAAI conference on artificial intelligence*. 2018.
2. Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." *nature*
 518.7540 (2015): 529-533.
3. Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." *Proceedings of the AAAI conference on artificial intelligence*
. Vol. 30. No. 1. 2016.
4. Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." *International conference on machine learning*
. PMLR, 2016.
5. Fortunato, Meire, et al. "Noisy networks for exploration." *arXiv preprint arXiv:1706.10295*
 (2017).
6. Schaul, Tom, et al. "Prioritized experience replay." *arXiv preprint arXiv:1511.05952*
 (2015).
7. Bellemare, Marc G., Will Dabney, and Rémi Munos. "A distributional perspective on reinforcement learning." *International Conference on Machine Learning*
. PMLR, 2017.
8. Hernandez-Garcia, J. Fernando, and Richard S. Sutton. "Understanding multi-step deep reinforcement learning: a systematic study of the DQN target." *arXiv preprint arXiv:1901.07510*
 (2019).