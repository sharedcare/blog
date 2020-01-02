---
layout: post
title:  "Introduction to Reinforcement Learning!"
date:   2020-01-02 19:35:42 +0800
categories: tutorial
---
### Chapter I (第一章): Teach AI to Play Games(教AI玩游戏)
强化学习(以下简称RL)在游戏中有着独特的优势。在这一章节，通过完成游戏AI来深入了解RL的原理。

**Following Projects:**

以下是一些通过RL制作的游戏AI项目，可供参考:

- [贪吃蛇][snake-ga]

#### Part I (第一部分): Simulate Basic Game Environment(模拟简单的游戏环境)
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

[gym]: https://gym.openai.com/
[snake-ga]: https://github.com/maurock/snake-ga
