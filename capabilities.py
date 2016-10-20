from agent import Agent

"""
This file contains stub implementations of AI capabilities.

To build an actual AI system, replace these stubs with real implementations.
"""

class Imitator(Agent):

    def __init__(self, expert, **kwargs):
        """
        expert: an Agent implementing the policy-to-be-imitated

        The Imitator will sometimes call expert.act() in order to get training data

        We promise that the number of calls to expert.act() will be sublinear
        in the number of calls to Imitator.act().

        Note that each Agent has immutable state,
        but calling methods on an Imitator may cause updates to external parameters,
        and these parameters may affect the behavior of existing Agent objects
        """
        super(Imitator, self).__init__(**kwargs)
        self.expert = expert

    #the act method is responsible for sometimes calling expert.act() to gather training data
    #it is also responsible for updating the agent's parameters

class IntrinsicRL(Agent):

    def __init__(self, reward, **kwargs):
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
        super(IntrinsicRL, self).__init__(**kwargs)
        self.reward = reward

    #the act method is responsible for sometimes calling reward() to gather training data,
    #and for updating the agent's parameters

class HybridLearner(Imitator, IntrinsicRL):
    """
    Combines Imitator and IntrinsicRL
    """
    pass

class TransparentRL(IntrinsicRL):
    """
    Extends IntrinsicRL by having the agent supply auxiliary data to the reward function
    """

    def __init__(self, info_reward, **kwargs):
        """
        info_reward: takes as input a sequence of observations, actions, and auxiliary information,
        returns a reward that measures the quality of the auxiliary information.

        the agent prooduces auxiliary info that approximately maximize info_reward, taking the actions as given

        this info will be supplied as an additional argument to the reward function;
        the actions will approximately maximize the reward function, taking the actions as given
        """
        super(TransparentRL, self).__init__(**kwargs)
        self.info_reward = info_reward

    #the act method sometimes collects supervised training data
    #to do so it computes the auxiliary information (using the internal state of the agent),
    #provides this information to reward(),
    #and then updates the models which generate the actions and the auxiliary information

class ComparisonRL(IntrinsicRL):

    """
    The reward function should now take a pair of (observation, action) tuples, and return value in [-1, 1].
    Negative values correspond to the first option having higher reward,
    while positive values correspond to the second value having higher reward.
    The reward should be antisymmetric.

    The agent's goal can be understood as playing a two-player game against itself:
    the agent's trajectories optimize its relative reward against the agent's distribution over trajectories,
    holding that distribution fixed---i.e, the agent should converge to the minimax strategy
    in this zero-sum game played against itself
    """
    pass

class TransparentComparisonRL(ComparisonRL, TransparentRL):
    """
    Combines TransparentRL and ComparisonRL

    info_reward should now take as input a sequence of observations, sequence of actions,
    and a pair of candidate pieces of auxiliary information.

    The auxiliary information will be optimized in the same way as for ComparisonRL

    The auxiliary information will be supplied to the reward function,
    which should now operate on a pair of tuples of (observations, actions, info).
    """
    pass

class ThrottledAgent(Agent):

    def __init__(self, capability, **kwargs):
        """
        capability: an integer representing the capability of the agent,
        which might control the model complexity, computational complexity, or training time

        We promise that agents with capability 0 are strictly weaker than a huamn,
        and that agents with capability n+1 are strictly weaker than amplify.amplify(A)
        for an agent of capability n.
        """
        self.capability = capability
        super(ThrottledAgent, self).__init__(**kwargs)

class PowerfulAgent(ThrottledAgent, TransparentComparisonRL, Imitator):
    """
    This is our model of a powerful learning system.
    It differs from a `generic' RL system in a few respects:

    * It can learn from any combination of imitation and reinforcement
    * Its reward function compares pairs of trajectories rather than evaluating individual trajectories
    * Its capability is explicitly controlled the capability= argument to the constructor
    * Its reward function is given auxiliary information which is
    produced by the agent to optimize a special info_reward function
    """
    pass
