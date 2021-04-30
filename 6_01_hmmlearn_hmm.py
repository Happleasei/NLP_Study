import numpy as np
from hmmlearn import hmm

states = ["晴", "雨", "阴"]
n_states = len(states)

observations = ["运动", "工作", "玩乐", "购物"]
n_observations = len(observations)

start_probability = np.array([0.3, 0.2, 0.5])
transition_probability = np.array([
    [0.7, 0.1, 0.2],
    [0.2, 0.6, 0.2],
    [0.3, 0.3, 0.4],
])

emission_probability = np.array([
    [0.4, 0.1, 0.4, 0.1],
    [0.3, 0.5, 0.1, 0.1],
    [0.3, 0.3, 0.3, 0.1]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_= start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

actions = np.array([[1, 0, 1, 1, 2, 3, 2]]).T
_, weathers = model.decode(actions, algorithm="viterbi")

print("行为:", ", ".join(map(lambda x: observations[int(x)], actions)))
print("天气:", ", ".join(map(lambda x: states[x], weathers)))
