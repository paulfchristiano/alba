import pyparsing as pp
from utils import unweave, areinstances
from agent import Agent, Budgeter, BudgetedAgent
from amplify.message import Message, Pointer, Channel, Referent, addressed_message

def HCH(H, n=int(1e8)):
    return Budgeter(BudgetedHCH(H), n)

class BudgetedHCH(BudgetedAgent):
    """
    HCH transforms an Agent that operates on text
    into a better-resourced BudgetedAgent that operates on messages.
    The total bandwidth of HCH(H) is limited by the bandwidth of H.
    """

    def __init__(self, H, child_base=None, args=()):
        self.H = H
        self.args = args 
        #by default, children are copies of self
        self.child_base = self if child_base is None else child_base
        assert self.well_formed()

    def child(self):
        return self.child_base

    def well_formed(self):
        return (
            isinstance(self.H, Agent) and
            isinstance(self.child_base, BudgetedHCH) and
            areinstances(self.args, Referent)
        )

    def act(self, obs, budget):
        state = self
        while True:
            message = state.view_message(obs)
            if budget < 0:
                raise Exception("It really shouldn't be possible to get to < 0 budget.")
            elif budget == 0:
                message += "\n[You have no budget, type a message to reply]"
            else:
                message += "\n[Remaining budget is {}]".format(budget)
            response, new_H = state.H.act(message)
            state = BudgetedHCH(new_H, state.child_base, state.args + obs.args)
            if budget <= 0:
                return parse_message(response), state, 0
            command = parse_command(response)
            obs, done, return_value, spending = command.execute(state, budget)
            budget -= spending
            if done: return return_value, state, budget

    def view_message(self, message):
        n = len(self.args)
        k = message.size
        return message.format_with_indices(range(n, n+k))

#----commands

class Command(object):

    def execute(self, env):
        raise NotImplemented()

class Ask(Command):

    def __init__(self, message, budget=None, recipient=None):
        self.message = message
        self.recipient_channel = recipient
        self.budget = budget

    def execute(self, env, budget):
        default_budget = budget / 10
        max_budget = budget - 1
        sub_budget = min(max_budget, self.budget if self.budget is not None else default_budget)
        message = addressed_message(env, self.message.instantiate(env.args))
        if self.recipient_channel is None:
            recipient = env.child()
        else:
            recipient = self.recipient_channel.instantiate(env.args).agent
        response, recipient, remaining = recipient.act(message, sub_budget)
        return addressed_message(recipient, response), False, None, sub_budget-remaining + 1

class View(Command):

    def __init__(self, message):
        self.message = message

    def execute(self, env, budget):
        return self.message.instantiate(env.args), False, None, 1

class Return(Command):

    def __init__(self, message):
        self.message = message

    def execute(self, env, budget):
        return None, True, self.message.instantiate(env.args), 1

class Reflect(Command):

    def execute(self, env, budget):
        return Message("you are []", Channel(env)), False, None, 1

class MalformedCommand(Command):

    def execute(self, env, budget):
        return Message("the valid commands are 'reply', 'ask', 'reflect', 'view', and 'ask@N'"), False, None, 1

#----parsing

def parse_command(s):
    try:
        return command.parseString(s, parseAll=True)[0]
    except pp.ParseException:
        return MalformedCommand()

def parse_message(s):
    try:
        return message.parseString(s, parseAll=True)[0]
    except pp.ParseException:
        return Message("<<malformed message>>")

def raw(s):
    return pp.Literal(s).suppress()

number = pp.Word("0123456789").setParseAction(lambda t : int(t[0]))
prose = pp.Word(" ,!?+-/*.;:_<>=&%{}[]\'\"" + pp.alphas).leaveWhitespace()

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

target_modifier = raw("@")+number
target_modifier.setParseAction(lambda xs : ("recipient", Pointer(xs[0], type=Channel)))

budget_modifier = raw("$")+number
budget_modifier.setParseAction(lambda xs : ("budget", xs[0]))

ask_modifiers = pp.ZeroOrMore(target_modifier ^ budget_modifier)
ask_modifiers.setParseAction(lambda xs : dict(list(xs)))

ask_command = (raw("ask")) + ask_modifiers + message
ask_command.setParseAction(lambda xs : Ask(xs[1], **xs[0]))

reply_command = (raw("reply") | raw("return")) + message
reply_command.setParseAction(lambda xs : Return(xs[0]))

reflect_command = raw("reflect")
reflect_command.setParseAction(lambda xs : Reflect())

view_command = raw("view") + message
view_command.setParseAction(lambda xs : View(xs[0]))

command = ask_command | reply_command | reflect_command | view_command
