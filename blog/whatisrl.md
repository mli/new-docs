
# What is Reinforcement Learning anyways?

Reinforcement learning has recently made the headlines on a number of occasions. One of the most prominent example is through AlphaGo, a software agent developed by DeepMind that beat the top Go human players. The latest variant, named simply AlphaZero, is not only a master at Go, but is also able to achieve superhuman performance on games like Chess, Poker and Shogi.

![The top human player, Ke Jie, in a match against DeepMind’s AlphaGo.](https://cdn-images-1.medium.com/max/5492/1*fHoW0qFN0ci4n9CaPVQuOQ.jpeg)*The top human player, Ke Jie, in a match against DeepMind’s AlphaGo.*

Another captivating result comes from the [Dota 2](https://www.theverge.com/2018/8/6/17655086/dota2-openai-bots-professional-gaming-ai) bot, which was developed by a team of researchers at OpenAI, leading to a victory against a team of former professional players. The two approaches have many subtle differences, but they are tied by the same underlying theory: reinforcement learning.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/UZHTNBMAfAA" frameborder="0" allowfullscreen></iframe></center>

The main advantage of reinforcement learning over supervised learning is the fact that it does not require labelled data, or more generally a *teacher*. This is great news, as in many applications it is simply infeasible to gather labelled data. More broadly, the path to Artificial General Intelligence can only come from self-learning agents that learn to generalize across different tasks, as we couldn’t provide examples of good behavior to the agent for every possible scenario. As such, reinforcement learning seems to be a good candidate. However, as promising as this avenue might seem, there are many obstacles in the way. Let’s start at the very beginning.

The basic idea of reinforcement learning is as follows: an agent interacts with an environment through actions. It receives as feedback a reward and the next state of the environment. Through this interactive loop, the agent learns what actions lead to high reward. For example, in the next figure we have a software agent learning to play the game Breakout. The observation (or state) available to the agent is the pixels of the screen. The agent processes this image through a convolutional neural network and then outputs an action believed to be optimal. In response to this action, the agent receives a reward, and the next observation. Through the reward signal, the agent will then modify its policy — it will learn how to be better.

![Reinforcement learning algorithms try to maximize a reward signal by interacting with an environment.](https://cdn-images-1.medium.com/max/2988/1*gU0UeTGfjghQSO-trsMSyg.png)*Reinforcement learning algorithms try to maximize a reward signal by interacting with an environment.*

As we notice, we haven’t made any assumptions about the environment, therefore reinforcement learning holds the promise of solving any environment: this could range from robotics applications, playing Atari games and up to sequence prediction. As we can expect, this means we will run into all sorts of complexities. One of the most important challenge is the exploration vs exploitation dilemma.

### The Exploration vs. Exploitation dilemma

Consider the situation where a reinforcement learning agent has to solve a navigation task in an environment with four rooms. The agent starts in the top-left room. In the same room, there is a gold coin that gives a reward of 1. At the same time, there is a key in the next room that opens a chest situated in the last room which contains a diamond, giving a reward of 100.

![FourRooms domain with two different rewards: an easy-to-find gold coin and a diamond in a chest.](https://cdn-images-1.medium.com/max/3372/1*uWyDCUTFUGWI50TX5oTQ6w.png)*FourRooms domain with two different rewards: an easy-to-find gold coin and a diamond in a chest.*

Figuring out how to get the diamond reward is a very difficult task as the sequence of actions leading to the diamond is very unlikely to be discovered. Not only is it difficult to explore these distant rewards, but the proximity of the gold coin will act as a gravitational force and constantly attract the agent. This comes from the fact that reinforcement learning agents usually do not have access to the stationary distribution of the task: this means that they can only train on samples/visited states. Given that the gold coin will be collected many times, this leads to an skewed training set. This situation is formulated through the exploration vs exploitation dilemma: we cannot do both at the same time, yet successful agents need do both through the learning process. To be able to explore more difficult areas of the game, an agent usually needs to master simple tasks through exploitation. However, to exploit the simple tasks, the agent must have explored them in the first place. Therefore, how do we decide when to perform exploration or exploitation?

As an analogy for supervised learning, we could imagine that our training set does not contain samples from a certain class of objects. Given a test set based on the complete distribution of the data, we would expect poor performance on the previously unseen classes. In this sense, reinforcement learning and supervised learning are two very different problems.

In today’s world, supervised learning is a widely applied technique, which is yet to be seen in the case of reinforcement learning. This comes once again from the fundamental dilemma of exploration vs. exploitation, which *underlies most problems in reinforcement learning*. Indeed, for the case of self-driving cars, we simply could not let a reinforcement learning agent smash vehicles until it understands how to drive.

That being said, reinforcement learning is a great solution if your problem has the two following characteristics: an easy way to get **tons of data** and a **clearly defined reward signal**. If an agent has the possibility to generate great amounts of data, it will be better at exploring actions and their effects (i.e. a long period of bumping around). The process can be improved if a well-defined reward signal is given to the agent. Indeed, reinforcement learning algorithms care only about one thing: the reward. They do not have an understanding of what the goal of the game might be. As we can see in the following video, an agent playing CoastRunners has discovered an usual strategy to maximize the total score (bottom left corner of the video).

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/tlOIHko8ySg" frameborder="0" allowfullscreen></iframe></center>

In this example, the reward function contains a fundamental flaw (getting bonus points for catching the turbo pack) which the agent learns to exploit. Therefore, a simple reward signal is your best recipe for success. However, until you try it, you simply do not know how the agent will react.

Despite these requirements, reinforcement learning is still being applied today in:

* Energy management : Google’s data center cooling has been [reduced](https://www.technologyreview.com/s/611902/google-just-gave-control-over-data-center-cooling-to-an-ai/) by 40% through reinforcement learning.

* Robotics: traditional techniques have long overshadowed reinforcement learning, but there has a been a recent surge of [interest](https://www.technologyreview.com/s/609339/these-ai-hotshots-plan-to-reboot-manufacturing-by-jumping-inside-robots/) in the field of manufacturing.

* Deep learning: interestingly, some of the most interesting applications exploiting the power of [supervised learning](https://arxiv.org/abs/1611.01578).

So far in this overview, we have seen the good and the bad of reinforcement learning. It is a promising avenue, but since the goal it is solving is complex, the path won’t be simple and will still take many years. After this general introduction, we will now proceed to implementing one of the most important reinforcement learning algorithms.

## **Implementing a reinforcement learning algorithm**

The field of reinforcement learning is full trade-offs: exploration vs exploitation, bias vs variance, online vs offline updates, model-based vs model-free… We will restrict our attention to one of the most successful approaches: the **REINFORCE** algorithm. Before we dig into it, let’s lay out some definitions.

Reinforcement learning is based on the Markov property assumption, that is, the task we are trying to solve is a **Markov Decision Process** (MDP). This implies that given an environment state S, we are able to determine what action A the agent should take, independently of states before S. After taking an action A in state S, the agent receives a reward R. This goes on until the end of an episode, after which we start anew from one of the states.

![A Markov Decision Process consists of states, actions, transition probabilities and rewards.](https://cdn-images-1.medium.com/max/3056/1*eGr8VgPojCU9de-1T5C9Mw.png)*A Markov Decision Process consists of states, actions, transition probabilities and rewards.*

This loop is defined simply in a few lines with the help of OpenAI [Gym’s](https://gym.openai.com/) interface:

<iframe src="https://medium.com/media/1c503bf60e6ff053a02ee5121b404098" frameborder=0></iframe>

An agent will follow a policy π(a | s), which outputs the actions’ probabilities for a certain state. This policy is in general obtained through function approximation, i.e. a neural network with weights θ. When using MXNet, we define our network in the following way:

<iframe src="https://medium.com/media/a6e7a8812cb6ccceadb3ac2ef089cac8" frameborder=0></iframe>

Our goal will be to improve an initial policy in order to obtain more rewards from the environment. To do this, the objective we will maximize is the expected discounted return (defined as G at time t):

![](https://cdn-images-1.medium.com/max/4440/1*ycSvlXLVQcOLUQPodLlNkw.png)

As we see, the return G_t is defined as the sum of future rewards R, each being multiplied by γ, the discount factor. This scalar acts as a tradeoff between immediate rewards (γ=0) and long-term rewards (γ=1). Our goal is to update the policy’s weight in order to choose better actions. We will not proceed to the full derivation here, but we will simply show that the update rule takes this final form :

![](https://cdn-images-1.medium.com/max/3648/1*jrpwo8_R9GH4paUUHVpDpw.png)

When we write this in MXNet, it takes the following form:

<iframe src="https://medium.com/media/92a4cfcb42c3d630c0e97a8073a56717" frameborder=0></iframe>

Interestingly, this has a similar interpretation to the cross-entropy loss used usually in the context of supervised learning:

![](https://cdn-images-1.medium.com/max/2496/1*ffan3rYUJ0g4gaCZajbWGw.png)

We notice some small differences : the expectation is replaced by a sum and that we add negative sign since it is a loss, but otherwise the formulas are equivalent. Importantly, in the supervised learning case, we have the correct label for a datapoint X. However, in the case of reinforcement learning, we have to collect the return G_t after taking action A in state S. This means that if we took an action somewhere in the middle of the episode, if we end up with a positive return, this action will be encouraged in the future. We might ask ourselves if this is really efficient, as many actions do not have a real effect on the outcome of a game. For example, if we kick a soccer ball and score, the actions taken between the moment we hit the ball and the moment when the ball enters the goal are not responsible for the positive outcome. As you could have guessed it, this is an active area of [research](https://www.reddit.com/r/MachineLearning/comments/8sq0jy/rudder_reinforcement_learning_algorithm_that_is/).

We put these ideas together in a simple [200-line](https://github.com/mklissa/REINFORCE) MXNet script (to be fair, most of it is comments explaining the code) that will run in a few minutes on CPU, performing your first successful learning through reinforcement learning! The environment we will be solving is the classic CartPole task, where the agent has to learn to balance a pole sitting on a cart by moving the cart left and right.

![The classic CartPole-v1 environment.](https://cdn-images-1.medium.com/max/2000/1*EudtyG8NQcv_IV75QlmoCQ.gif)*The classic CartPole-v1 environment.*

That should do it! After running the code, you should have a learning curve similar to this one:

![Plot of the average rewards per episode: this is the combination of 5 different seeds.](https://cdn-images-1.medium.com/max/7500/1*zEolVGDnL_EnvHG_tUcOHA.png)*Plot of the average rewards per episode: this is the combination of 5 different seeds.*

### Conclusion

The algorithm we have implemented is called REINFORCE: it is one of the first algorithms in a long series of algorithms using the policy gradient theorem. Today’s state-of-the-art has improved in many different ways. However, it is still possible to solve some of the MuJoCo environments using REINFORCE. Only minor modifications are necessary in order to account for the continuous action space, as actions define the amount of force applied to the robot’s joints.

![Some of the MuJoCo environments](https://cdn-images-1.medium.com/max/2000/1*v5NJxQVQ4DBkoRobNuoCwQ.gif)*Some of the MuJoCo environments*

However, compared to more recent algorithms, such an implementation would take a long time to learn and would be limited in terms of capacity (i.e. wouldn’t solve some of the more complex environments).

Looking ahead, there are many implementations of state-of-the-art algorithms such as [A3C](https://arxiv.org/abs/1602.01783), [TRPO](https://arxiv.org/abs/1502.05477) and [PPO](https://arxiv.org/abs/1707.06347). On this matter, the researchers at [OpenAI](https://github.com/openai/baselines) have done an amazing job by open-sourcing their high-quality implementations. If you are looking to experiment further, this is the place to start!
