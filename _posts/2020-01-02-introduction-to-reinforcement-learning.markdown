---
title:  "Introduction to Reinforcement Learning!"
date:   2020-01-02 19:35:42 +0800
categories: tutorial
comment_id: 4
published: false
---

### Chapter I (第一章): Teach AI to Play Games(教AI玩游戏)
强化学习(以下简称RL)在游戏中有着独特的优势。在这一章节，通过完成游戏AI来深入了解RL的原理。
Google's DeepMind在这一领域中有很多研究，例如AlphaGo的最终版本[AlphaZero][alpha-zero]以及称霸StarCraft II的[AlphaStar][alpha-star]。
OpenAI在RL方面也有诸多贡献，尤其是开源的[游戏环境][openai-env]以及他们最近在[Multi-Agent][openai-multi-agent-paper]上的研究成果。

{% include image.html url="/assets/imgs/2020-01-02-introduction-to-reinforcement-learning/rl_diagram.png" description="经典的强化学习结构" %}

进一步了解[Q-learning][q-learning]以及它的变种[Deep Q-learning][deep-ql]和[Double Q-learning][double-ql]

**Following Projects:**

以下是一些通过RL制作的游戏AI项目，可供参考:

- [贪吃蛇][snake-ga]
- [躲猫猫][hide-and-seek]
- [Mario][mario]

#### Part I (第一部分): Create Basic Game Environment(模拟简单的游戏环境)
RL是基于环境而进行决策，所以模拟游戏环境将是最基础且复杂的第一步。
利用OpenAI的Gym工具包，可以通过python代码简单实现模拟一些类似Atari的简单游戏。
{% highlight python %}
import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
{% endhighlight %}
<iframe width="560" height="315" src="https://www.youtube.com/embed/J7E6_my3CHk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

#### Part II (第二部分): Reward Function(奖励规则)

#### References:

[Gym-OpenAI][gym]

[DeepMind][deepmind]


[gym]: https://gym.openai.com/
[deepmind]: https://deepmind.com/blog
[q-learning]: https://en.wikipedia.org/wiki/Q-learning
[deep-ql]: https://arxiv.org/pdf/1704.03732.pdf
[double-ql]: https://papers.nips.cc/paper/3964-double-q-learning.pdf
[snake-ga]: https://github.com/maurock/snake-ga
[hide-and-seek]: https://openai.com/blog/emergent-tool-use/
[mario]: http://pastebin.com/ZZmSNaHX
[openai-env]: https://gym.openai.com/
[openai-multi-agent-paper]: https://arxiv.org/pdf/1909.07528.pdf
[alpha-zero]: https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
[alpha-star]: https://deepmind.com/blog/announcements/deepmind-and-blizzard-open-starcraft-ii-ai-research-environment

