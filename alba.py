from capabilities import PowerfulAgent
from amplify import amplify, stateless_amplify
from memoizer import Memoizer

#FIXME: Prevent catastrophic failure on adversarial inputs. Adversarial training?

#TODO: Implement PowerfulAgent: semi-supervised RL+imitation with comparisons
#TODO: Implement throttling so that A -> Meta(HCH(A)) gets us from capability=n to n+1,
#and determine whether this is actually compatible with having capability increase unboundedly
#TODO: Implement effective transparency for PowerfulAgent,
#and determine whether this actually allows the overseer to evaluate the agent's actions
#TODO: Analyze amplify and ensure that it actually preserves reliability
#(figure out whether and how much A -> Ensemble((A, A, A)) actually amplifies reliability,
#and how much A -> Meta(HCH(A)) decreases reliability)
#TODO: More effectively elicit preferences from overseer, in particular ensuring that the mechanism
#appropriately handles the agent's uncertainty about the overseer's rating
#(perhaps compare to a fixed basket to measure strength of preference?)

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

def distill(overseer, n):
    expert = overseer
    reward, info_reward = make_comparisons(overseer)
    return PowerfulAgent(expert=expert, reward=reward, info_reward=info_reward, capability=n)

ALBA = make_ALBA(distill=distill, amplify=lambda A,H,n: amplify(A))

#This isn't a serious algorithm, but it's useful for messing around and testing.
#Rather than imitating in an intelligent way, it just copies what it has seen done before,
#and defers to the overseer whenever it encounters a novel situation.
memoizer_ALBA = make_ALBA(distill=lambda O,n: Memoizer(O), amplify=lambda A,H,n: stateless_amplify(A))

#---reward functions

#NOTE: this code isn't actually tested
#I've mocked up an imitation agent but not an RL agent

def str_to_reward(s):
    return min(1, max(-1, float(s)))

def make_transcript(observations, actions):
    lines = interleave(
            [str(obs) for obs in observations],
            ["{}\n".format(act) for act in actions]
        )
    return "\n".join(lines)


def make_rewards(overseer):
    def reward(observations, actions, explanation=None):
        transcript = make_transcript(observations, actions)
        message = "Assign a score between -1 and 1 to the following transcript:\n\n{}\n".format(transcript)
        if explanation is not None:
            message += (
                "\nYour evaluations can make use of the following auxiliary information"
                "provided by the agent:\n\n{}\n".format(explanation)
            )
        response, _ = overseer.act(message)
        return str_to_reward(response)

    def info_reward(observations, actions, explanation):
        transcript = make_transcript(observations, actions)
        message = (
            "Assign a score between -1 and 1 to the following auxiliary information...\n\n{}\n\n"
            "...as a description of the following transcript:\n\n{}".format(explanation, transcript)
        )
        response, _ = overseer.act(message)
        return str_to_reward(response)

    return reward, info_reward

def make_comparisons(overseer):
    def compare(tuple1, tuple2):
        observations1, actions1, info1 = tuple1
        observations2, actions2, info2 = tuple2
        message = (
            "Compare the following two transcripts, and output a score between -1 and +1; "
            "a score of -1 indicates the first transcript is significantly better, +1 indicates that "
            "the second transcript is significantly better. After each transcript will be some explanatory "
            "information which might help you make a better decision."
            "\n\nTranscript 1:\n\n{}\n\nExplanation 1:\n\n{}"
            "\n\nTranscript 2:\n\n{}\n\nExplanation 2:\n\n{}\n\n".format(
                make_transcript(observations1, actions1), info1,
                make_transcript(observations2, actions2), info2
            )
        )
        response, _ = overseer.act(message)
        return str_to_reward(response)

    def compare_info(observations, actions, info1, info2):
        message = (
            "Compare the following two pieces of info as an explanation of the subsequent transcript,"
            "and output a score between -1 and +1,"
            "where -1 and indicates the first explanation is better and +1 indicates the second is better..."
            "\n\nExplanation 1:\n\n{}"
            "\n\nExplanation 2:\n\n{}"
            "\n\nTranscript:\n\n{}".format(
                info1, info2, make_transcript(observations, actions)
            )
        )
        response, _ = overseer.act(message)
        return str_to_reward(response)

    return compare, compare_info
