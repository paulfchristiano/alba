# alba
ALBA is a candidate scheme for training an aligned reinforcement learner, as originally outlined
in [this post](https://medium.com/ai-control/alba-an-explicit-proposal-for-aligned-ai-17a55f60bbcf).
A more recent blog post is [here](https://medium.com/@paulfchristiano/5636ef510907).

For now the system is theoretical; the point of writing this code (and running it with mocked up
AI capabilities) is to be very concrete/precise about exactly how the proposed system
would work. Over the medium or long term we might have an implementation that is practical enough to run,
and then we could learn something about what works and what doesn't.

For now the system has a number of deal-breaking problems (see the TODO and FIXME's in alba.py),
such that it definitely wouldn't be aligned if you used it with powerful AI.
Over the long term these issues will either get addressed,
or it will become clear that ALBA is unworkable.

The implementation of HCH is very similar to the interpreter I wrote
[here](https://github.com/paulfchristiano/dwimmer), though I didn't reuse any code from that project.

## requirements

You'll need a few packages:
```
pip install pymongo
pip install pyparsing
pip install six
```

If you want to use memoizer_ALBA
then you need to have a mongo server running locally. This is pretty easy to set up,
but there isn't that much to see.

## usage

The best place to start is to look in examples.py.

Agents have the single method act(observation) which returns an action and a new Agent.
To simulate a conversation between A and B you would run something like:

```
A1 = Agent1()
A2 = Agent2()
message1, A1 = A1.act("start off the conversation")
while True:
  message2, A2 = A2.act(message1)
  message1, A1 = A1.act(message2)
``` 

The project is largely organized as a calculus of agents;
methods like amplify.HCH(A) or capabilities.Imitator(A) turn one agent into another agent.

If you want to get a sense for what the algorithm looks like from inside,
you can run examples.py and then try meta.act("this is a test") or ensemble.act("this is a test").
 
## what's in the box

* capabilities.py defines stand-ins for powerful AI capabilities:
IntrinsicRL, Imitator, [HybridLearner](https://medium.com/ai-control/imitation-rl-613d70146409),
[TransparentRL](https://medium.com/ai-control/the-informed-oversight-problem-1b51b4f66b35),
and TransparentHybrid.
* alba.ALBA(H, n) returns a new agent which is aligned with H but much smarter,
roughly as defined [here](https://medium.com/ai-control/alba-an-explicit-proposal-for-aligned-ai-17a55f60bbcf). It won't actually work unless you replace capabilities.TransparentHybrid with an actual algorithm for [imitation+RL](https://medium.com/ai-control/imitation-rl-613d70146409). Its inputs and outputs are strings.
* agent.Human() is the agent implemented by the user sitting at their computer. Its inputs and outputs are strings.
* amplify.HCH(A) roughly implements [annotated functional programming]
(https://medium.com/ai-control/approval-directed-algorithm-learning-bf1f8fad42cd). Its inputs and outputs are Messages.
* amplify.amplify(A) is an implementation of [capability amplification](https://medium.com/ai-control/policy-amplification-6a70cbee4f34) that turns A into a more powerful (but slower) agent. The current implemenation is just Meta(HCH(A)).
* memoizer.Memoizer(A) is a very simple "learning" algorithm that tries to memorize what A does,
and asks A whenever it encounters a novel situation.
* alba.memoizer_ALBA(H, n) is like ALBA, but defined using memoizer.Memoizer instead of a real learning algorithm. This one will actually work, but good luck getting it to do anything.

## using HCH

If you run alba.memoizer_ALBA(H, n) you will be presented with a message asking you what to do.
You can type commands, and those commands will get executed.
This is the machinery in the definition of HCH,
and it works roughly as described in [the post](https://medium.com/ai-control/strong-hch-bedb0dc08d4e#.6s6wcmyqu).

You operate in an environment, which consists of a sequence of messages.
Each message consists of text interleaved with:

* Pointers like `#5` to messages
* Pointers like `@2` to agents

You can compose a message by writing text interspersed with `#n` or `@n`.
You can also insert submessages using `(...)`; for example you could write `the pair with first element (the city of Paris) and second element (the country France)`.
You can't write numbers. It's very easy to accidentally type `n` instead of `#n` or `@n`, I know the UI could use improvement.
But like I said above, there isn't much to see.

You can interact with this environment by typing commands.
The available commands are:

* `view #n` for an integer n: `view #3` will show you the message pointed to by `#3`. You are guaranteed that at most one pointer in the environment will be rendered as `#3`, so this is always unambiguous.
* `ask Q` for a message Q: `ask what is #1 plus #2?` will create a new agent, start it off with the message `what is #1 plus #2?` run it until it replies, and then returns whatever it replied. Here `#1` and `#2` would be pointers to messages, which the agent could view using the `view` command. They might have different names in the new agent's environment.
* `reply A` for a message A: `reply the answer is #4` will return `the answer is #4` as your reply to whichever agent initiated the current interaction by using `ask` or `ask@`.
* `ask@n Q` for an integer n and message Q: `ask@1 what do you mean?` will send the message `what do you mean?` to the agent pointed to by `@1`, run that agent until it replies, and then return whatever it replies.
* `reflect`: returns a pointer to the agent which calls reflect. These pointers can be inserted into a message, e.g. if you were asked `who are you?` you could call `reflect`, receive the reply `you are @1`, and then run `reply I am @1`.

When you receive a message or reply, it will begin with `@n:...`.
You can use the pointer `@n` to address queries to the agent who sent that message.
Remember that everything is immutable. If I send a message `Q` to agent `@7`,
the reply will come from a new agent `@8`.
If I send another message `Q2` to `@7`, it will have no memory of having just answered `Q`.
If I want to continue the discussion, I need to address the next message to `@8`.

If you don't like what the system has memorized and want to start over, you can run `mongo` and then at the shell type `db.memoizer.remove({})` and that will destroy everything.
