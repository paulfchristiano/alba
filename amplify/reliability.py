from collections import defaultdict
from agent import Agent

def amplify_reliability(A):
    return Ensemble((A, A, A))

def unzip(xs):
    xs = tuple(xs)
    return tuple(tuple(x[i] for x in xs) for i in range(len(xs[0])))

def dialog(agents, initial_message):
    agents = list(agents)
    N = len(agents)
    to = ["(You are {}. Once you have decided, say 'done'.)\n{}".format(i, initial_message) for i in range(N)]
    done = [False for i in range(N)]
    while not all(done):
        for i in range(N):
            if not done[i]:
                message, agents[i] = agents[i].act(to[i])
                to[i] = ""
                if message == "done":
                    done[i] = True
                message = "{} says: {}".format(i, message)
                for j in range(N):
                    to[j] = "{}\n{}".format(to[j], message) if to[j] else message
    return tuple(agents)

def run_election(votes):
    vote_counts = defaultdict(lambda : 0)
    for vote in votes:
        for v in vote.split(","):
            try:
                vote_counts[int(v)] += 1
            except ValueError:
                pass
    winner = None
    best_count = -1
    for option, vote_count in vote_counts.items():
        if vote_count > best_count:
            best_count = vote_count
            winner = option
    return winner

class Ensemble(Agent):

    def __init__(self, agents):
        self.agents = agents

    def act(self, observation):
        agents = self.agents
        intro = (
            "Discuss amongst yourselves and decide what you want to do. "
            "The next observation will be displayed after this."
        )
        intro += "\n\n{}".format(observation)
        agents = dialog(agents, intro)
        actions, agents = unzip(agent.act("what action do you propose?") for agent in agents)
        vote_query = (
            "which of these proposals do you approve of? "
            "enter the numbers one by one, separated by commas"
        )
        summary = "\n".join("{} proposes: {}".format(i, action) for i, action in enumerate(actions))
        message = "{}\n\n{}".format(vote_query, summary)
        votes, agents = unzip(agent.act(message) for agent in agents)
        pick = run_election(votes)
        _, agents = unzip(agent.act("action {} won".format(pick)) for agent in agents)
        return actions[pick], Ensemble(agents)
