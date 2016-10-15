from capabilities import Imitator, IntrinsicRL, HybridLearner, TransparentHybrid
from amplify import HCH
from amplify.meta import Meta, StatelessMeta
from memoizer import Memoizer

def make_ALBA(distill, amplify=lambda A, H : Meta(HCH(A))):
    """
    distill: takes as input an expensive agent,
    uses it to train a cheap learning agent
    amplify: takes as input a weak agent and a human,
    uses it to produce a slow but powerful agent
    """
    def ALBA(H, n):
        overseer = H if n == 0 else amplify(ALBA(H, n-1), H)
        return distill(overseer)
    return ALBA

ALBA = make_ALBA(lambda overseer : TransparentHybrid(overseer, make_reward(overseer)))

#TODO: Build powerful AI
#TODO: Implement semi-supervised RL
#TODO: Implement transparent RL

#FIXME: Prevent catastrophic failure on adversarial inputs. Adversarial training?
#FIXME: Ensure RL agent cannot outsmart overseer. Gradually scale up capacity as a function of n?
#FIXME: Prevent failure probability from growing with each iteration. Amplify reliability as well as capability?
#FIXME: Allow Amplify(A) to learn from training data, so it can keep up with the RL agent it is overseeing
#FIXME: Scores in [0, 1] are arbitrary, use comparisons between different actions instead
#FIXME: Use budgeted HCH so that errors can't result in hangs

#TODO: Figure out whether iterating A -> Meta(HCH(A)) can really get us to arbitrarily powerful agents

#---simple examples

imitation_ALBA = make_ALBA(lambda overseer : Imitator(overseer))
rl_ALBA = make_ALBA(lambda overseer : IntrinsicRL(make_reward(overseer)))
hybrid_ALBA = make_ALBA(lambda overseer : HybridLearner(overseer, make_reward(overseer)))

#This isn't a serious algorithm, but it's useful for messing around and testing
#Rather than imitating in an intelligent way, it just copies what it has seen done before,
#and defers to the overseer whenever it encounters a novel situation.
#NOTE: memoizer_alba requires a stateless agent, so that memoization can work correctly
memoizer_ALBA = make_ALBA(Memoizer, lambda A, H : StatelessMeta(HCH(A)))

#---reward functions

def str_to_reward(s):
    return min(1, max(0, float(s)))

def make_reward(overseer):
    def reward_fn(observations, actions, explanation=None):
        lines = interleave(
                [str(obs) for obs in observations],
                ["{}\n".format(act) for act in actions]
            )
        transcript = "\n".join(lines)
        message = "Assign a score between 0 and 1 to the following transcript:\n\n{}\n".format(transcript)
        if explanation is not None:
            message += "\nYour evaluations can make use of the following auxiliary information provided by the agent:\n\n{}\n".format(explanation)
        response, overseer = overseer.act(message)
        reward = str_to_reward(response)
        if explanation is not None:
            info_message = "Now assign a score between 0 and 1 to the auxiliary information, based on its usefulness"
            response, overseer = overseer.act(info_message)
            info_reward = str_to_reward(response)
            return reward, info_reward
        else:
            return reward

    return reward_fn
