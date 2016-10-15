import pyparsing as pp
from utils import unweave, areinstances
from agent import Agent
from amplify.message import Message, Pointer, Channel, Referent, addressed_message

class HCH(Agent):
    """
    HCH transforms an agent that operates on text
    into a better-resourced agent that operates on messages.
    The total bandwidth of HCH(H) is limited by the bandwidth of H.
    """

    def __init__(self, H, child_base=None, args=()):
        self.H = H
        self.args = args 
        #by default, children are copies of self
        self.child_base = self if child_base is None else child_base
        #assert self.well_formed()

    def child(self):
        return self.child_base

    def well_formed(self):
        return (
            isinstance(self.H, Agent) and
            isinstance(self.child_base, HCH) and
            areinstances(self.args, Referent)
        )

    def act(self, obs):
        state = self
        while True:
            instruction, new_H = state.H.act(state.view_message(obs))
            state = HCH(new_H, state.child_base, state.args + obs.args)
            command = parse_command(instruction)
            obs, done, return_value = command.execute(state)
            if done: return return_value, state

    def view_message(self, message):
        n = len(self.args)
        k = message.size
        return message.format_with_indices(range(n, n+k))

#----commands

class Command(object):

    def execute(self, env):
        raise NotImplemented()

class Ask(Command):

    def __init__(self, message, recipient=None):
        self.message = message
        self.recipient_channel = recipient

    def execute(self, env):
        message = addressed_message(env, self.message.instantiate(env.args))
        if self.recipient_channel is None:
            recipient = env.child()
        else:
            recipient = self.recipient_channel.instantiate(env.args).agent
        response, recipient = recipient.act(message)
        return addressed_message(recipient, response), False, None

class View(Command):

    def __init__(self, message):
        self.message = message

    def execute(self, env):
        return self.message.instantiate(env.args), False, None

class Return(Command):

    def __init__(self, message):
        self.message = message

    def execute(self, env):
        return None, True, self.message.instantiate(env.args)

class Reflect(Command):

    def execute(self, env):
        return Message("you are []", Channel(env)), False, None

class MalformedCommand(Command):

    def execute(self, env):
        return Message("the valid commands are 'reply', 'ask', 'reflect', 'view', and 'ask@N'"), False, None

#----parsing

def parse_command(s):
    try:
        return command.parseString(s, parseAll=True)[0]
    except pp.ParseException:
        return MalformedCommand()

def raw(s):
    return pp.Literal(s).suppress()

number = pp.Word("0123456789").setParseAction(lambda t : int(t[0]))
prose = pp.Word(" ,!?+-/*.;:_<>=&%${}[]\'\"" + pp.alphas).leaveWhitespace()

agent_referent = (raw("@")+ number).leaveWhitespace()
agent_referent.setParseAction(lambda x : Pointer(x[0], Channel))

message_referent = (raw("#") + number).leaveWhitespace()
message_referent.setParseAction(lambda x : Pointer(x[0], Message))

message = pp.Forward()
submessage = raw("(") + message + raw(")")
argument = submessage | agent_referent | message_referent
literal_message = (
        pp.Optional(prose, default="") +
        pp.ZeroOrMore(argument + pp.Optional(prose, default=""))
    ).setParseAction(lambda xs : Message(tuple(unweave(xs)[0]), *unweave(xs)[1]))
message << (message_referent ^ literal_message)

ask_command = raw("ask ") + message
ask_command.setParseAction(lambda xs : Ask(xs[0], recipient=None))

ask_at_command = (raw("ask@") + number) + message
ask_at_command.setParseAction(lambda xs : Ask(xs[1], recipient=Pointer(xs[0], type=Channel)))

reply_command = (raw("reply") | raw("return")) + message
reply_command.setParseAction(lambda xs : Return(xs[0]))

reflect_command = raw("reflect")
reflect_command.setParseAction(lambda xs : Reflect())

view_command = raw("view") + message
view_command.setParseAction(lambda xs : View(xs[0]))

command = ask_at_command | ask_command | reply_command | reflect_command | view_command
