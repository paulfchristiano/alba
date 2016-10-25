from amplify.message import Message
from agent import Agent, StatelessAgent

class Meta(Agent):
    """
    Meta transforms an agent that operates on messages
    into an agent that operates on text,
    by using messages to implement an algorithm operating on text.
    The total bandwidth of Meta() is generally unlimited.
    """

    def __init__(self, agent, state=None):
        self.state = Message("the initial state of an agent") if state is None else state
        self.agent = agent

    def act(self, obs):
        obsm = encode_str(obs)
        agent = self.agent
        query = Message("what string should be returned by an agent in state [] who observes []?", self.state, obsm)
        actionm, agent = agent.act(query)
        state, agent = agent.act(Message("what should the agent's state be after responding?"))
        return decode_str(actionm, self.agent), Meta(self.agent, state)

def StatelessMeta(agent):
    """
    Like Meta, but without state
    """
    return StatelessAgent(stateless_meta_policy(agent))

def stateless_meta_policy(agent):
    def policy(observations, actions):
        observationsm = encode_list([encode_str(obs) for obs in observations])
        actionsm = encode_list([encode_str(act) for act in actions])
        message = (
            "what string should an agent output after observing the sequence of observations [], "
            "given that their past responses have been []?"
        )
        query = Message(message, observationsm, actionsm)
        actionm, _ = agent.act(query)
        return decode_str(actionm, agent)
    return policy


#-----conversions between python objects and Messages

def decode_str(x, H):
    agent = H
    char_list, agent = agent.act(Message("what is the list of characters in []?", x))
    return "".join([decode_char(x, H) for x in decode_list(char_list, H)])

def encode_str(x):
    char_list = encode_list([encode_char(c) for c in x])
    return Message("the string with list of characters []", char_list)

def decode_char(x, H):
    agent = H
    code, agent = agent.act(Message("what is the ASCII code of []?", x))
    return chr(decode_int(code, H))

def encode_char(x):
    code = encode_int(ord(x))
    return Message("the character with ASCII code []", code)

def decode_list(x, H):
    agent = H
    empty, agent = agent.act(Message("is [] empty? (respond yes/no)", x))
    if empty == Message("yes"):
        return []
    elif empty == Message("no"):
        x, agent = agent.act(Message("what is its first element?"))
        xs, agent = agent.act(Message("what is the list of remaining elements?"))
        return [x] + decode_list(xs, H)
    else:
        raise Exception("is empty is not 'yes' or 'no'")

def encode_list(x):
    if len(x) == 0:
        return Message("the empty list")
    else:
        return Message("the list with first element [] and remaining elements in the list []", x[0], encode_list(x[1:]))

def decode_int(x, H):
    agent = H
    signm, agent = agent.act(Message("is [] negative, zero, or positive? (respond verbatim)", x))
    if signm == Message("negative"):
        sign = -1
        x = -x
    elif signm == Message("positive"):
        sign = 1
    elif signm == Message("zero"):
        return 0
    else:
        raise Exception("sign is not 'negative', 'positive', or 'zero'")
    paritym, agent  = agent.act(Message("is it even or odd? (respond verbatim)"))
    if paritym == Message("even"):
        parity = 0
    elif paritym == Message("odd"):
        parity = 1
    else:
        raise Exception("Parity is not 'even' or 'odd'")
    half, agent = agent.act(Message("what is half of it, rounded towards zero?"))
    return sign * (2 * decode_int(half, H) + parity)

def encode_int(x):
    if x == 0:
        return Message("zero")
    elif x < 0:
        return Message("the additive inverse of []", encode_int(-x))
    elif x%2 == 0:
        return Message("two times []", encode_int(x/2))
    elif x%2 == 1:
        return Message("two times [] plus one", encode_int(x/2))
    raise Exception()

def decode_float(x, H):
    agent = H
    A, agent = agent.act(Message("represent [] as A * 2^B; what is A?", x))
    B, agent = agent.act(Message("and what is B?"))
    return decode_int(A, H) * 1.0 / 2**decode_int(B, H)
