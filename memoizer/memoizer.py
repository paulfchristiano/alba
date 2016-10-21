from agent import Agent
import hashlib
import pymongo
import six
from utils import interleave, unweave

def hashable(x):
    return isinstance(x, six.string_types)

def md5(x):
    if isinstance(x, six.string_types):
        return hashlib.md5(x).digest().encode("base64")
    raise Exception("unhashable type {}".format(type(x)))

def hash_append(h, x):
    """
    h: a hash representing a list
    x: an object to be appended to that list
    """
    return md5(h + md5(x))

def hash_empty():
    """
    returns a hash representing an empty list
    """
    return md5("")

class MongoCache(object):

    def __init__(self, name="memoizer"):
        self.db = pymongo.MongoClient().cache
        if name not in self.db.collection_names():
            self.db.create_collection(name)
        self.collection = self.db[name]

    def lookup(self, key):
        resp = self.collection.find_one({"key":key})
        if resp is None or "value" not in resp:
            return None
        return resp["value"]

    def save(self, key, value):
        self.collection.update_one({"key":key}, {"$set":{"value":value}}, upsert=True)

#TODO generalize Memoizer to handle arbitrary Agents with serializable state,
#implement agent hashing so that this is computationally efficient for HCH

class Memoizer(Agent):

    def __init__(self, agent, cache=None, transcript=(), transcript_hash=None):
        self.transcript = transcript
        if transcript_hash is None:
            transcript_hash = hash_empty()
            for x in self.transcript: transcript_hash = hash_append(transcript_hash, x)
        self.transcript_hash = transcript_hash
        self.cache = cache or MongoCache()
        self.agent = agent 
        assert self.well_formed()

    def well_formed(self):
        return (
            isinstance(self.transcript, tuple) and
            all(hashable(x) for x in self.transcript) and
            isinstance(self.transcript_hash, six.string_types) and
            isinstance(self.agent, Agent) and
            self.agent.state_free
        )

    def extend(self, x):
        transcript = self.transcript + (x,)
        transcript_hash = hash_append(self.transcript_hash, x)
        return Memoizer(self.agent, self.cache, transcript, transcript_hash)

    def set(self, observations, actions):
        return Memoizer(self.agent, self.cache, interleave(observations, actions))

    def lookup(self):
        return self.cache.lookup(self.transcript_hash)

    def save(self, action):
        return self.cache.save(self.transcript_hash, action)

    def act(self, obs):
        new = self.extend(obs)
        act = new.lookup()
        if act is None:
            act = self.agent.set(*unweave(self.transcript)).act(obs)[0]
            new.save(act)
        return act, new.extend(act)
