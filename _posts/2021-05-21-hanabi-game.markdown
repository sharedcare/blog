---
title:  "Hanabi Game: A Board Game with Potential Challenges"
date:   2021-05-21 00:00:00 +0800
categories: intro
description: Recommend a fun and promising board game - Hanabi. Unlike other common card games, it requires cooperation rather than competition among all players
comment_id: 4
---

# Intro

Hanabi(花火) originally means "fireworks" in Japanese. As a common Japanese cultural symbol, it was used by a French game designer Antoine Bauza to create a forward-thinking card game. Unlike most of the popular competitive card games on the market, Hanabi game is designed as a cooperative game. 

The rules of the game are simple, yet cleverly constructed to create a complete system. The core of the game lies in the information, but no one has the complete information. So this game has to be played as a group, with all players really and truly committed to the game. This may seem simple, but in reality many cooperative games fail to do this. In some games it is easy for one player to direct the other players to carry out the action. In Hanabi, players are oonly allowed to exchange very little information which means the players do not have all the knowledge available to make a good decision. Players cannot see their own cards and they have to infer the intentions of other players through limited hints. That's why the Hanabi game is unique as a imperfect information cooperative game.

## Game Rules

This is a 2-5 player cooperative game where players try to create the perfect fireworks by placing the cards on the table in the correct order. In Hanabi, there are five sequences of cards in different colors. The goal of the game is to collect the following 5 groups of colors of fireworks and to build five card sequences.

<div style="margin: auto;" align="center">
    <img class="image image--xl" src="/assets/imgs/2021-05-21-hanabi-game/hanabi.jpg"/> 
</div>

What makes the game interesting is that players can only see their teammates' cards, but not their own. Communication happens in large part through the 'cue' action, where one person tells the other about their cards so they know what to play or discard. Since a limited number of cues can be provided, good players can communicate strategically and utilize conventions such as "discard the oldest card first".

### Components

- 50 Hanabi cards
    - 5 colors (white, red, blue, yellow, green)
    - the values on the cards to be dealt are 1, 1, 1, 2, 2, 3, 3, 4, 4, 5 for each color
- 8 Blue tokens
- 3 Red tokens

### Set Up

- Place the 8 blue tokens in the box and the 3 red tokens out of the box
- Shuffle the 50 cards to make a deck and put them face down. Deal a hand of cards to each player
    - For 2 or 3 players, deal 5 cards to each player
    - For 4 or 5 players, deal 4 cards to each player

<aside>
⚠️ The players should not look at the cards which are dealt to them during the entire game

</aside>

### Game Play

In a player's turn, he must complete one of the following three actions(without skipping the turn):

1. **Giving a piece of information**
    - In order to carry out this task, the player has to take a blue token from the box and place near the red tokens. He can then tell a teammate something about the cards that this player has in his hand
    
    <aside>
    ⚠️ The player must clearly point to the cards which he is giving the information about
    
    </aside>
    
    - **Two types of information can be given**
        - Information about only one color
        - information about a value
        
        <aside>
        💡 The player must give complete information - If a player has two green cards, the informer cannot only point to one of them
        
        </aside>
        
    
    <aside>
    ⚠️ If there is none of blue tokens in the box then the player has to perform another action
    
    </aside>
    
2. **Discarding a card**
    - Performing this task allows a blue token to be returned to the box. The player discards a card from his hand and puts it in the discard pile. Then he takes a new card and adds it to his hand without looking at it
    
    <aside>
    ⚠️ This action cannot be performed if all the blue tokens are in the box. The player has to perform another action
    
    </aside>
    
3. **Playing a card**
    - The player takes a card from his hand and puts it in front of him. Two options are possible:
        - The card either begins or completes a firework and it is then added to this firework
        - Or the card does not complete any firework, it is then discarded and a red token is added into the box
        
        Then the player takes a new card and adds it to his hand without looking at it
        

### End of the Game

- There are 3 ways to end the game of Hanabi:
    - The game ends immediately and is lost if the third red token is added into the box
    - The game ends immediately and it is a stunning victory if the firework makers manage to make the 5 fireworks before the cards run out. The players are then awarded the maximum score of 25 points
    - The game ends if a firework maker takes the last card from the pile. Then each player plays one more time, including the player who picked up the last card. The players cannot pick up cards during this final round as the pile is empty

## Communicating while playing Hanabi
- Communication between the players is very essential to Hanabi Game. If you strictly follow the rules, the communication between you and your teammates is limited. However, we can analyze the intention of our teammates to provide information by looking at all known cards on the field and the previous hints.

# Approach
There are two major settings of the game: 

1. Self-play
    - AI plays with copies of itself and therefore it knows a better about its teammates
2. Ad-hoc teams
    - A set of agents need to cooperate that are not familiar with each other (including human players)

## Learning Agents

How about the existing reinforcement learning methods for Hanabi? In fact, not as well as one might expect

### Actor-Critic-Hanabi-Agent

- Asynchronous advantage actor-critic algorithms
- Importance Weighted Actor-Learner variant to address the stale gradient problem

### Rainbow-Agent

- Deep Q-Network
- Multi-agent version of Rainbow based on the Dopamine framework
- The agents control the different players share parameters

### BAD-Agent

- Bayesian Action Decoder
- Use a bayesian belief
- All agents track a public belief including common knowledge about the cards

## Rule-Based Approaches

Unlike the previous reinforcement learning methods, these rule-based strategies directly encode conventions through their rules

### SmartBot

- Play/Discard strategy
- Tracks the publicly known information about each player’s cards
- Do not works in a ad-hoc team setting

### HatBot

- Recommendation and information strategy
- Uses a predefined protocol to determine a recommended action for all other players when giving hints
- 5-player Hanabi

### WTFWThat

- Information strategy
- 2-5 players

### FireFlower

- Human-style conventions
- Tracks both private and common knowledge
- 2-ply search over all possible actions with a modeled probability distribution over its partner's expected response

# Conclusion

It is clear that machine learning has incredible potential considering the results of the last decade. The next big step for AI will be for intelligences to learn to communicate and reasoning intent.

Similar to how the Atari 2600 game has inspired the field of deep reinforcement learning, Hanabi game is a great environment for testing how algorithms learn to cooperate in scenarios that are simple for humans but more challenging for AI.

In this case, we can use AI to understand this 'theory of mind' technique to not only play this game well, but also to apply it to a broader range of cooperative tasks - especially those with humans.

## Why Hanabi Card Game is the next step?

This project helps AI to learn new ways to communicate with each other effectively. The goal is to teach the AI to learn communication and work together towards a common goal. Through this Hanabi game as a proxy, we can finally reuse the algorithm for other real world applications where communication and cooperation with human is essential.

# References

1. N. Bard et al., “The Hanabi Challenge: A New Frontier for AI Research,” Artificial Intelligence, vol. 280, p. 103216, Mar. 2020, doi: 10.1016/j.artint.2019.103216.
2. J. N. Foerster et al., “Bayesian Action Decoder for Deep Multi-Agent Reinforcement Learning,” arXiv:1811.01458 [cs], Sep. 2019, Accessed: May 21, 2021. [Online]. Available: [http://arxiv.org/abs/1811.01458](http://arxiv.org/abs/1811.01458)
3. M. Hessel et al., “Rainbow: Combining Improvements in Deep Reinforcement Learning,” arXiv:1710.02298 [cs], Oct. 2017, Accessed: Jun. 11, 2021. [Online]. Available: [http://arxiv.org/abs/1710.02298](http://arxiv.org/abs/1710.02298)
4. C. COX, J. D. SILVA, P. DEORSEY, F. H. J. KENTER, T. RETTER, and J. TOBIN, “How to Make the Perfect Fireworks Display: Two Strategies for Hanabi,” Mathematics Magazine, vol. 88, no. 5, pp. 323–336, 2015, doi: 10.4169/math.mag.88.5.323.
5. “GitHub - Quuxplusone/Hanabi: Framework for writing bots that play Hanabi.” [https://github.com/Quuxplusone/Hanabi](https://github.com/Quuxplusone/Hanabi) (accessed Jun. 11, 2021).
6. “GitHub - lightvector/fireflower: A rewrite of hanabi-bot in Scala.” [https://github.com/lightvector/fireflower](https://github.com/lightvector/fireflower) (accessed Jun. 11, 2021).
7. “GitHub - WuTheFWasThat/hanabi.rs: State of the art Hanabi bots + simulation framework in rust.” [https://github.com/WuTheFWasThat/hanabi.rs](https://github.com/WuTheFWasThat/hanabi.rs) (accessed Jun. 11, 2021).