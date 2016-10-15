from utils import elicit_input

class Agent(object):

    #an immutable representation of a (stateful) policy

    def act(self, obs):
        """
        returns (action, state of Agent after computing action)
        """
        raise NotImplementedError

    def done(self):
        """
        if this is a learning agent,
        calling done() may trigger additional updates to the agent's
        """
        pass

    @property
    def state_free(self):
        """
        set allows us to set the agent to the state it would occupy if it had seen
        a particular sequence of observations and actions

        set is only available on agents that don't maintain internal state

        this property is used when we want to assert that an agent is state free
        """
        return hasattr(self, 'set')

class StatelessAgent(Agent):

    def __init__(self, policy, observations=(), actions=()):
        """
        policy: a map from (observations, actions) to next action

        returns: the agent who uses this policy
        """
        self.observations = observations
        self.actions = actions
        self.policy = policy

    def act(self, obs):
        observations, actions = self.observations, self.actions
        observations += (obs,)
        action = self.policy(observations, actions)
        actions += (action,)
        return action, StatelessAgent(self.policy, observations, actions)

    def set(self, observations, actions):
        return StatelessAgent(self.policy, observations, actions)

Human = StatelessAgent(elicit_input)
