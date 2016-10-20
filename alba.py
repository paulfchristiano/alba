from capabilities import PowerfulAgent, SecondBestAgent
from amplify import amplify, stateless_amplify
from memoizer import Memoizer

#TODO: Build powerful AI
#TODO: Implement semi-supervised RL
#TODO: Implement transparent RL

#FIXME: Prevent catastrophic failure on adversarial inputs. Adversarial training?
#FIXME: Allow Amplify(A) to learn from training data, so it can keep up with the RL agent it is overseeing.
#FIXME: Scores in [0, 1] are arbitrary. Use PowerfulAgent instead of SecondBestAgent.
#FIXME: Use budgeted HCH so that errors can't result in hangs.

#TODO: Analyze amplify and ensure that it actually preserves reliability
#(figure out whether and how much A -> Ensemble((A, A, A)) actually amplifies reliability,
#and how much A -> Meta(HCH(A)) decreases reliability)
#TODO: Implement throttling so that A -> Meta(HCH(A)) gets us from capability=n to n+1
#(figure out whether this is actually compatible with having capability increase unboundedly)

def make_ALBA(distill, amplify):
    """
    distill: takes as input an expensive agent,
    uses it to train a cheap learning agent
    amplify: takes as input a weak agent and a human,
    uses it to produce a slow but powerful agent
    """
    def ALBA(H, n):
        overseer = H if n == 0 else amplify(ALBA(H, n-1), H, n)
        return distill(overseer, n)
    return ALBA

def second_best_distillation(overseer, n):
    expert = overseer
    reward, info_reward = make_rewards(overseer)
    return SecondBestAgent(expert=expert, reward=reward, info_reward=info_reward, capability=n)

ALBA = make_ALBA(distill=second_best_distillation, amplify=lambda A,H,n: amplify(A))

#This isn't a serious algorithm, but it's useful for messing around and testing.
#Rather than imitating in an intelligent way, it just copies what it has seen done before,
#and defers to the overseer whenever it encounters a novel situation.
memoizer_ALBA = make_ALBA(distill=lambda O,n: Memoizer(O), amplify=lambda A,H,n: stateless_amplify(A))

#---reward functions

#NOTE: this code isn't actually tested
#I've mocked up an imitation agent but not an RL agent

def str_to_reward(s):
    return min(1, max(0, float(s)))

def make_rewards(overseer):
    def reward(observations, actions, explanation=None):
        lines = interleave(
                [str(obs) for obs in kbservations],
                ["{}\n".format(act) for act in actions]
            )
        transcript = "\n".join(lines)
        message = "Assign a score between 0 and 1 to the following transcript:\n\n{}\n".format(transcript)
        if explanation is not None:
            message += (
                "\nYour evaluations can make use of the following auxiliary information"
                "provided by the agent:\n\n{}\n".format(explanation)
            )
        response, overseer = overseer.act(message)
        return str_to_reward(response)

    def info_reward(observations, actions, explanation):
        lines = interleave(
                [str(obs) for obs in kbservations],
                ["{}\n".format(act) for act in actions]
            )
        transcript = "\n".join(lines)
        message = (
            "Assign a score between 0 and 1 to the following auxiliary information...\n\n{}\n\n"
            "...as a description of the following transcript:\n\n{}".format(explanation, transcript)
        )
        response, overseer = overseer.act(message)
        return str_to_reward(response)

    return reward, info_reward
