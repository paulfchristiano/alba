def areinstances(xs, t):
    return isinstance(xs, tuple) and all(isinstance(x, t) for x in xs)

def interleave(xs, ys):
    assert len(xs) == len(ys) + 1 or len(xs) == len(ys)
    result = []
    for x, y in zip(xs, ys):
        result.append(x)
        result.append(y)
    if len(xs) > len(ys):
        result.append(xs[-1])
    return tuple(result)

def unweave(xs):
    result = ([], [])
    for i, x in enumerate(xs):
        result[i%2].append(x)
    return tuple(result[0]), tuple(result[1])

def clear_screen():
    print("\x1b[2J\x1b[H")

def elicit_input(observations, actions):
    clear_screen()
    lines = interleave(observations, [">>> {}".format(action) for action in actions])
    print("\n\n".join(lines))
    return raw_input("\n>>> ")
