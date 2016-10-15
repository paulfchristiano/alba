from agent import Agent

"""
This file contains stub implementations of AI capabilities.

To build an actual AI system, replace these stubs with real implementations.
"""

class Imitator(Agent):

    def __init__(self, expert):
        """
        expert: an Agent implementing the policy-to-be-imitated

        The Imitator will sometimes call expert.act() in order to get training data

        We promise that the number of calls to expert.act() will be sublinear
        in the number of calls to Imitator.act().

        Note that each Agent has immutable state,
        but calling methods on an Imitator may cause updates to external parameters,
        and these parameters may affect the behavior of existing Agent objects
        """
        self.expert = expert

    def act(self, obs):
        #this method is responsible for sometimes calling expert.act() to gather training data
        raise NotImplementedError

    def done(self):
        #this method actually updates the agent's parameters if it gathered training data
        #the agent might also update on episodes where it doesn't gather training data
        raise NotImplementedError

class IntrinsicRL(Agent):

    def __init__(self, reward):
        """
        reward: a function that acts on a pair of observation and action sequences,
        and returns a reward in [0, 1]

        The agent will sometimes call reward() in order to evaluate a partial trajectory.
        Its goal is to maximize the total reward
        of all of the complete trajectories produced by calling act().
        We say that a trajectory is complete if act() was never called
        on the agent's state at the end of the trajectory.

        Note that each Agent has immutable state,
        but calling methods on an RL agent may cause updates to external parameters,
        and these parameters may affect the behavior of existing Agent objects
        """
        self.reward = reward

    def act(self, obs):
        raise NotImplementedError

    def done(self):
        #this method is responsible for sometimes calling reward() to gather training data,
        #and for updating teh agent's parameters
        raise NotImplementedError

class HybridLearner(Agent):
    """
    Combines Imitator and IntrinsicRL
    """

    def __init__(self, expert, reward):
        """
        Will call expert.act() and reward() a sublinear number of times,
        and aims to get a reward as high as the expert.
        """
        self.expert = expert
        self.reward = reward

    def act(self, obs):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

class TransparentRL(Agent):
    """
    Extends IntrinsicRL by having the agent supply auxiliary data to the reward function
    """

    def __init__(self, rewards):
        """
        rewards: takes as input a sequence of observations, actions, and auxiliary information, and
        returns two numbers, action_reward and info_reward

        info_reward is an evaluation of the auxiliary information to evaluating the sequence of actions
        action_reward is an evaluation of the agent's behavior, in light of the auxiliary information

        the auxiliary_information will approximately maximize info_reward, taking the actions as given
        while the actions will approximately maximize the reward (as for an IntirinsicRL agent), taking the info as given
        """
        self.rewards = reward

    def act(self, obs):
        raise NotImplementedError

    def done(self):
        #this method sometimes collects supervised training data
        #to do so it computes the auxiliary information (using the internal state of the agent),
        #provides this information to reward(),
        #and then updates the models which generate the actions and the auxiliary information
        raise NotImplementedError

class TransparentHybrid(Agent):
    """
    Combines TransparentRL with Imitator, analogous to HybridLearner
    """

    def __init__(self, expert, rewards):
        self.expert = expert
        self.rewards = rewards

    def act(self, obs):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

