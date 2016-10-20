#Human is the agent implemented by you
from agent import Human

#this will raise NotImplementedError owing to how we don't have an AGI yet
#(also it definitely wouldn't really be aligned)
from alba import ALBA
putative_aligned_AI = ALBA(Human, 20)

#this is a simplified version that consults the expert whenever it encounters a novel situation
#(the memoization mechanism also prevents the overseer from maintaining state)
#to run it you will need to have a mongo server running locally
from alba import memoizer_ALBA
mocked_up_AI = memoizer_ALBA(Human, 1)

#if you want to see how HCH works, you can run hch.act()
from amplify import Meta, HCH
hch = Meta(HCH(Human))

#if you want to see how the ensembling works, you can run council.act()
from amplify.reliability import Ensemble
council = Ensemble((Human, Human))
