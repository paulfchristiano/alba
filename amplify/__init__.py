from hch import HCH
from meta import Meta, StatelessMeta
from reliability import amplify_reliability

hch_budget = int(1e8)

def amplify(A):
    return Meta(HCH(amplify_reliability(A), hch_budget))

def stateless_amplify(A):
    return StatelessMeta(HCH(amplify_reliability(A), hch_budget))
