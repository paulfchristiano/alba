from utils import elicit_input

class Agent(object):

    #an immutable representation of a (stateful) policy

    def act(self, obs):
        """
        returns (action, state of Agent after computing action)
        """
        raise NotImplementedError("Agents must define act")

    @property
    def state_free(self):
        """
        set allows us to set the agent to the state it would occupy if it had seen
        a particular sequence of observations and actions

        set is only available on agents that don't maintain internal state

        this property is used when we want to assert that an agent is state free
        """
        return hasattr(self, 'set')

    def __call__(self, obs):
        return self.act(obs)

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

class BudgetedAgent(object):
    """
    Like an Agent, but tracks resource constraints.
    """

    def act(self, obs, budget):
        """
        returns (action, state of Agent after computing action, remaining budget)
    
        The method tracks some kind of resources; it should only use resources in budget,
        and it should return any unused resources.
        """
        raise NotImplementedError("BudgetedAgents must define act")

class Budgeter(Agent):
    """
    Turns a BudgetedAgent into an Agent by specifying what its per-step budget should be.
    """

    def __init__(self, A, budget):
        self.A = A
        self.budget = budget
        
    def well_formed(self):
        return (
            isinstance(self.A, BudgetedAgent)
        )

    def act(self, obs):
        action, A, _ = self.A.act(obs, self.budget)
        return Budgeter(A, self.budget)
