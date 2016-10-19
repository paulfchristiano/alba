# alba
ALBA is a candidate scheme for training an aligned reinforcement learner, as originally outlined
in [this post](https://medium.com/ai-control/alba-an-explicit-proposal-for-aligned-ai-17a55f60bbcf).

For now the system is theoretical; the point of writing it as code (and running it with mocked up
AI capabilities) is to force ourselves to be extremely concrete about exactly how the proposed system
would work. Over the medium term this implementation will hopefully become practical enough to run,
and we can start to see what works and doesn't work.

For now the system has a number of deal-breaking problems (see the TODO and FIXME's in alba.py),
over the long term those will either get addressed or it will become clear that the scheme is unworkable.

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

The project is largely organized as a calculus of agents, supplying methods like amplify(A) that turn
one agent into another agent.
 
## what's in the box

* capabilities.py defines stand-ins for powerful AI capabilities:
IntrinsicRL, Imitator, [HybridLearner](https://medium.com/ai-control/imitation-rl-613d70146409),
[TransparentRL](https://medium.com/ai-control/the-informed-oversight-problem-1b51b4f66b35),
and TransparentHybrid.
* alba.ALBA(H, n) returns a new agent which is aligned with H but much smarter,
roughly as defined [here](https://medium.com/ai-control/alba-an-explicit-proposal-for-aligned-ai-17a55f60bbcf). It won't actually work unless you replace capabilities.TransparentHybrid with an actual algorithm for [imitation+RL](https://medium.com/ai-control/imitation-rl-613d70146409).
* agent.Human() is the agent implemented by the user sitting at their computer.
* amplify.HCH(A) roughly implements [strong HCH](https://medium.com/ai-control/strong-hch-bedb0dc08d4e)
* amplify.Meta(H) roughly implements [annotated functional programming]
(https://medium.com/ai-control/approval-directed-algorithm-learning-bf1f8fad42cd), 
though this runs together with HCH.
* memoizer.Memoizer(A) is a very simple "learning" algorithm that tries to memorize what A does,
and asks A whenever it encounters a novel situation.
* alba.memoizer_ALBA(H, n) is like ALBA, but defined using memoizer.Memoizer instead of a real learning algorithm.
