# Blackjack Agent
Treating the game of Blackjack as a Markov Decision Process, this research notebook attempts to train an agent to play the game using the Deep Q-Learning environment.

# Packages used
- `time`
- `collections`
- `gym`
- `numpy`
- `PIL`
- `tensorflow`
- `pyvirtualdisplay`
- `copy`


# Blackjack Environment
We will [OpenAI's Gym library](https://www.gymlibrary.dev/) to load and attempt to solve the Blackjack environment. 

The goal of the Blakcjack environment is to train an agent to beat the dealer in Blackjack by obtaining cards that sum close to 21, without going over 21, and yet still have a higher value thant the dealer's card.

<br>
<br>
<figure>
  <img src ="https://www.gymlibrary.dev/_images/blackjack.gif" width = 40%>
      <figcaption style = "text-align: center; font-style: italic">Blackjack-v1 Environment</figcaption>
</figure>

# Action Space
The action space consists of two actions represented by discrete values.
- `0`: Stick
- `1`: Hit

# Observation Space
The agent's observation space is a state vector containing 3 variables:
1. Player's current sum `[int]`
2. Dealer's one showing card (1- 10) `[int]`
3. Whether a player holds a usable ace `[bool]`

# Rewards
- Win game: +1
- Lose game: -1
- Draw: 0
- Win game with natural Blackjack: +1.5 if `natural=True`, else +1
