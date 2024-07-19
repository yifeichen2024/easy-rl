# MarkovDecisionProcess(MDP) for RL马尔科夫
## MDP Terminology 
1. **Agent智能体**：训练主要对象
2. **Environment环境**：与智能体进行交互的其他物体. 
3. **State状态**：智能体的当前的状态。对于机器人来说可以是他的位置姿态信息。
4. **Action动作**：智能体所作出的与时间相关的动作。
5. **Policy策略**： 做出动作的背后原因。是一个对于一系列动作的概率分布。

## Markov Property马尔科夫性质
A state $S_t$ is Markov if and only if
$$
    \mathbb{P}[S_{t+1} | S_{t}] = \mathbb{P}[S_{t+1 | S_1, S_2,...,S_t}]
$$
*Example*: 一个机器人的一系列动作：
1. 坐在椅子上
2. 站起来
3. 右腿向前
如果当前状态是右腿向前$S_t$，那这一状态是取决于他的前一状态$S_{t-1}$——站起来，而不会取决于再之前一步——坐在椅子上.

## Markov Process Explained 解释马尔科夫过程
$$
\mathcal{P_{ss'}} =  \mathbb{P}[S_{t+1} = s' | S_{t} = s]
$$
马尔科夫过程由`(S,P)`来定义。`s` 是状态，`P`是状态转换概率。这个过程包括一系列随机的状态`s_1, s_2`,...这些状态都遵循马尔科夫性质。
状态转换概率`P_{ss'}`是从当前`s'`转换到`s`的概率。例子如图所示。
![image](/docs/chapter1/image/3_markov-decision-process.png "markov-process")

## Markov reward Process 马尔科夫奖励过程
$$
\mathcal{P_{ss'}} =  \mathbb{P}[S_{t+1} = s' | S_{t} = s]
$$
$$
\mathcal{R_{s}} =  \mathbb{E}[R_{t+1} | S_{t} = s]
$$
马尔科夫奖励过程由`(S,P,R,y)`来定义。`S`为状态，`P`为状态转换概率，`R`是reward奖励,`y`是discount factor折扣因子。

`R_s`状态奖励是从`s_t`转换到所有可能状态的预期奖励。是在`s_t`这一状态而获得的。也就是当机器人离开之一状态后，才收获这奖励`R_{t+1}`
![image](/docs/chapter1/image/5_markov-decision-process.png "markov-process")

## Markov Decision Process(MDP) 马尔科夫决策
$$
\mathcal{P_{ss'}} =  \mathbb{P}[S_{t+1} = s' | S_{t} = s]
$$
$$
\mathcal{R_{s}} =  \mathbb{E}[R_{t+1} | S_{t} = s]
$$
一个马尔科夫决策过程(MDP)由`(S, A, P, R, y)`定义。$A$是一系列动作。在加入动作之前，MRP中的$P$和$R$都相对更多或更少随机。但是现在智能体本身可以通过动作来决定自己的状态

## Return $G_t$ 长期奖励回报
$$
G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum^\infty_{k=0}\gamma^k R_{t+k+1}
$$
$G_t$是每一步打折后的奖励加和。
对于RL来说，我们不想让智能体只注重当下时刻的奖励，而忽略掉长期的奖励。这个长期的奖励就是Return。

## Discount $\gamma$ 折扣因子
折扣因子$\gamma \in [0, 1]$。之所以引入折扣因子，是因为未来的奖励是不确定的。在考虑长期未来的奖励的时候，我们也要限制长期奖励带来的影响（对当下奖励的影响）

## Policy $\pi$ 策略
$$
\pi(a | s) = \mathbb{P}[A_t=a | S_t = s]
$$
$\pi$ 是动作对于状态的分布。策略定义了一个智能体的行为。
数学叙述是，他是在一个状态采取一个特定动作的概率。

## Value Functions价值函数
价值函数是一个状态或动作的长期价值。也就是说，它是一个状态或动作所带来的长期回报的期望。也是我们想优化的重点。
$$
v(s) = \mathbb[G_t | S_t = s]
$$
`v(s)` 是从状态`s`得到的期望长期回报

## Bellman Expectation Equation for Markov Reward Process 马尔可夫奖励过程的Bellman期望方程
Bellman期望方程是一个标准的价值函数表示形式。它将价值函数氛围两个部分：
1. 即使奖励 $R_(t+1)$
2. 打折的未来状态价值 $\gamma v(S_{t+1})$
![image](/docs/chapter1/image/10_markov-decision-process.png)
![image](/docs/chapter1/image/11_markov-decision-process.png)

例子如下：
![image](/docs/chapter1/image/12_markov-decision-process.png)
智能体从`s`转换到`s'`，价值函数为所有状态的期望回报。类似的可以用递归的方式将下一状态`s'`的长期回报替换为`s'`的价值函数。数学表达为：
$$
v(s)=\mathbb{E}[R_{t+1}+\gamma v(S_{t+1}) | S_t = s]
$$
求解方程上面的方程

期望独立分布，分别求解$R_{t+1}$ 和$v(s')$.已经得到了
$$
\mathcal{R_{s}} =  \mathbb{E}[R_{t+1} | S_{t} = s]
$$
而$v(s')$ 在所有$s'$的期望值由*Expected value*定义得到。

最终得到
$$
v(s) = \mathcal{R_{s}}+\gamma \sum_{s' \in S} \mathcal{P}_{ss'}v(s')
$$
总结来说，状态奖励是我们在状态`s`一定会收到的固定值，另一个就是所有其他状态`s'`的平均状态价值。

### State Value Function for Markov Decision Process (MDP)状态价值函数
$$
v_{\pi}(s)=\mathbb{E}_{\pi}[G_t | S_t=s]
$$
在状态$s$，使用策略$\pi$的长期回报的期望。
与马尔科夫奖励过程相似。
### Action Value Function for Markov Decision Process (MDP)动作价值函数
$$
q_{\pi}(s,a)=\mathbb{E}_{\pi}[G_t | S_t=s, A_t = a]
$$
将智能体的动作作为状态转换的参数之一。这个方程给的是对于动作的长期回报的期望。

### Bellman Expectation Equation (for MDP)
$$
v_{\pi}(s)=\mathbb{E}_{\pi}[R_{t+1}+\gamma v(S_{t+1}) | S_t = s]
$$
$$
q_{\pi}(s,a)=\mathbb{E}_{\pi}[R_{t+1}+\gamma v(S_{t+1}， A_{t+1}) | S_t = s, A_t = a]
$$
通过下图来理解。
![image](/docs/chapter1/image/18_markov-decision-process.png)
圆圈代表状态，点代表动作。左图代表以状态为核心，右图代表以动作为核心。
- **圈到点**：在一个状态，智能体按照策略采取了一个动作。
- **点到圈**：环境作用在智能体上，并根据转换概率将智能体送到另一状态。

将这两个过程看成两个很小的过程：
$$
v_{\pi}(s) = \sum_{a \in \mathcal A}\pi(a | s)q_{\pi}(s, a)
$$
因为是有状态到动作的这个转换，所以取期望动作价值over所有的动作。这完全符合Bellma方程。想同的是对于动作价值函数。
$$
q_\pi(s,a) = \mathcal{R_{s}^a}+\gamma \sum_{s' \in S} \mathcal{P}^{a}_{ss'}v_{\pi}(s')
$$
我们可以在状态价值函数中替换此方程，以获得类似于MRP的递归状态价值函数的值（反之亦然）：
$$
v_{\pi}(s) = \sum_{a \in \mathcal A}\pi(a | s)(\mathcal{R}^a_s+\gamma \sum_{s' \in S} \mathcal{P}^{a}_{ss'}v_{\pi}(s'))
$$
$$
q_\pi(s,a) = \mathcal{R_{s}^a}+\gamma \sum_{s' \in S} \mathcal{P}^{a}_{ss'} \sum_{a' \in \mathcal{A}}\pi(a' | s')q_\pi(s', a')
$$

### Markov Decision Process Optimal Value Functions马尔可夫决策过程最优值函数
如果我们能获得MDP中所有动作导致的所有的状态或动作的价值，那去价值最大的就可以了。
$$
v_*(s)=\max_\pi v_\pi(s)
$$
$$
q_*(s,a)=\max_\pi q_\pi(s,a)
$$
可以设定如下：
$$
\pi_*(a | s)= \left\{
\begin{matrix}
1 & \text{If }a=\argmax_{a \in \mathcal{A}}q_*(s,a)\\
0 & \text{otherwise}
\end{matrix}\right.
$$

### Bellman Optimality Equation

因为我们最终都会选择价值最大的动作，因此我们可以将这个Value设为优化价值函数：
$$
v_*(s)=\max_\pi q_*(s,a)
$$
$$
q_*(s,a) = \mathcal{R_{s}^a}+\gamma \sum_{s' \in S} \mathcal{P}^{a}_{ss'}v_{*}(s')
$$

上式不会有太大变化，因为这个部分是环境作用主导，智能体无法控制。然而因为我们遵循最优策略，状态价值函数会是最优的。

虽然说我们获得MDP所有状态动作的价值，我们就能得到最优，但是状态和动作的状态数百万种的，我们无法对其所有都进行评估。
因此这里只讨论了将RL问题用MDP表述，并在MDP下对智能体进行评估，却并没有探寻最优价值和策略的解。
