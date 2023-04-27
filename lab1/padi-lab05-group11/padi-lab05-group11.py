import numpy as np
import matplotlib.pyplot as plt

#Activity 1
def load_chain(filename):
    m = np.load(filename)
    states = []
    for i in range(0,len(m)):
        states.append(str(i))
    states = tuple(states)
    P = np.array(m)
    return states,P
    

#Activity 2
def prob_trajectory(chain, trajectory):
    prob = 1
    for i in range(0,len(trajectory)-1):
        prob = prob * chain[1][int(trajectory[i]),int(trajectory[i+1])]
    return prob

# Activity 3
def stationary_dist(chain):
    values, vectors = np.linalg.eig(chain[1].T)
    for i in range(len(values)):
        if(np.isclose(np.real(values[i]),np.real(1))):
            value = i
            break
    vector = vectors[:,value]
    sum = np.sum(vector)
    u_star = np.real(vector/sum)
    return u_star

# The probabilities in the distribution tell how much time the truck spends in each state, when the probabibility for a state is higher it means that the truck spent more time in this state.

# Activity 4
def compute_dist(chain, start, steps):
    matrixpred = np.linalg.matrix_power(chain[1],steps)
    res = start.dot(matrixpred)
    return res

# This chain is ergodic because after a large amount of steps we can observe that the chain converges to a stationary distribution. In this case 2000 steps.

# Activity 5
def simulate(chain, start, steps):
    state = np.random.choice(chain[0],1,False,p=start[0])
    res = [state[0]]
    for i in range(0,steps - 1):
        prob = chain[1][int(state)]
        state = np.random.choice(chain[0],1,False,p=prob)
        res.append(state[0])
    
    return res

# Activity 6
M = load_chain('garbage-big.npy')

nS = len(M[0])

u = np.ones((1, nS)) / nS

np.random.seed(42)

simulation = simulate(M, u, 50000)
traj = []
for i in simulation:
    traj.append(int(i)-0.5)
traj.append(len(M[0])-0.5)
plt.hist(traj, rwidth=0.7, density= True, bins=len(M[0]))
plt.plot(M[0], stationary_dist(M))




