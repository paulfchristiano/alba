from utils import areinstances, interleave, unweave
import six

class Referent(object):
    """
    A Referent is anything that can be referred to in a message,
    including Messages, Pointers, and Channels
    """

    symbol = "?"

    def instantiate(self, xs):
        raise NotImplemented()

class Message(Referent):
    """
    A Message consists of text interspersed with Referents
    """

    symbol = "#"

    def __init__(self, text, *args):
        if isinstance(text, six.string_types):
            text = tuple(text.split("[]"))
        args = tuple(args)
        self.text = text
        self.args = args 
        assert self.well_formed()

    def well_formed(self):
        return (
            areinstances(self.text, six.string_types) and
            areinstances(self.args, Referent) and
            len(self.text) == len(self.args) + 1
        )

    @property
    def size(self):
        return len(self.args)

    def __add__(self, other):
        joined = self.text[-1] + other.text[0]
        return Message(self.text[:-1] + (joined,) + other.text[1:], *(self.args + other.args))

    def format(self, names):
        return "".join(interleave(self.text, names))

    def format_with_indices(self, indices):
        return self.format(["{}{}".format(arg.symbol, index) for arg, index in zip(self.args, indices)])

    def __str__(self):
        return self.format(['({})'.format(arg) for arg in self.args])

    def __eq__(self, other):
        return self.text == other.text and self.args == other.args

    def __ne__(self, other):
        return self.text != other.text or self.args != other.args

    def instantiate(self, xs):
        return Message(self.text, *[arg.instantiate(xs) for arg in self.args])

class Channel(Referent):
    """
    A Channel is a wrapper around an Agent, that lets it be pointed to in messages
    """

    symbol = "@"

    def __init__(self, agent):
        self.agent = agent
        assert self.well_formed()

    def well_formed(self):
        return hasattr(self.agent, 'act')

    def instantiate(self, xs):
        raise Exception("should not try to instantiate a channel")

def addressed_message(sender, message):
    return Message("[]: ", Channel(sender)) + message

class Pointer(Referent):
    """
    A Pointer is an abstract variable,
    which can be instantiated given a list of arguments
    """


    def __init__(self, n, type=Referent):
        self.n = n
        self.type = type
        #assert self.well_formed()

    def well_formed(self):
        return (
            issubclass(self.type, Referent) and
            isinstance(self.n, int)
        )

    def instantiate(self, xs):
        x = xs[self.n]
        assert isinstance(x, self.type)
        return x

    @property
    def symbol(self):
        return "{}->".format(self.type.symbol)

    def __str__(self):
        return "{}{}".format(self.symbol, self.n)
