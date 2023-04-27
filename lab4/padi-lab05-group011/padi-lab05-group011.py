import numpy as np
import numpy.random as rnd

mdp_info = np.load('garbage.npz', allow_pickle=True)

# The MDP is a tuple (X, A, P, c, gamma)
M = mdp_info['M']

# We also load the optimal Q-function for the MDP
Qopt = mdp_info['Q']

# Activity 1
def sample_transition(mdp, s, a):
    states = mdp[0]
    probs = mdp[2]
    costs = mdp[3]
    
    cost = costs[s, a]
    
    new_state = rnd.choice(np.arange(len(states)), p=probs[a][s])
    return (s,a,cost,new_state)

# Activity 2
def egreedy(Q,eps=0.1):
    choice = rnd.choice([0,1], p=[eps, 1-eps])
    
    if choice == 0:
        return rnd.choice(len(Q))
    else:
        return rnd.choice(np.flatnonzero(Q == Q.min()))
    
# Activity 3
def mb_learning(mdp,n,qinit,Pinit,cinit):
    e = 0.15
    states = mdp[0]
    actions = mdp[1]
    gama = mdp[4]
    
    Q = qinit
    P = Pinit
    C = cinit
    
    N = np.zeros((len(states), len(actions)))
    state = rnd.choice(len(states)) 
    
    for i in range(n):
        action = egreedy(Q[state],e)
        N[state, action] += 1
        _, _, cnew, snew = sample_transition(mdp, state, action)
        
        alfa = 1/(N[state, action] + 1)
        P[action][state,:] *= 1-alfa
        C[state, action] = alfa*(C[state, action] - cnew) + cnew
        
        P[action][state,:][snew] += alfa
        Q[state, action] = C[state, action] + gama*(P[action][state,:].dot(np.amin(Q, axis=1,keepdims=True)))
        state = snew
        
    return (Q, P, C)

# Activity 4
def qlearning(mdp,n,qinit):
    e = 0.15
    states = mdp[0]
    gama = mdp[4]
    
    Q = qinit
    state = rnd.choice(len(states)) 
    alfa = 0.3
    
    for i in range(n):
        action = egreedy(Q[state],e)
        _, _, cnew, snew = sample_transition(mdp, state, action)
        
        Q[state, action] = Q[state, action] + alfa*(cnew + gama*np.amin(Q[snew] - Q[state, action]))
        state = snew
        
    return Q


# Activity 5
def sarsa(mdp,n,qinit):
    e = 0.15
    states = mdp[0]
    gama = mdp[4]
    
    Q = qinit
    state = rnd.choice(len(states)) 
    alfa = 0.3
    
    action = egreedy(Q[state],e)
    
    for i in range(n):
        _, _, cnew, snew = sample_transition(mdp, state, action)
        actionnew = egreedy(Q[snew],e)
        
        Q[state, action] = Q[state, action] + alfa*(cnew + gama*Q[snew, actionnew] - Q[state, action])
        state = snew
        action = actionnew
        
    return Q

# Activity 6.
# In model based learning we can see that it converges to a better solution much faster in terms of iterations than the other two. This happens because this model updates the Q-function and also updates the costs and probabilities matrixes, so we are making use of all the information we have. Even though this converges faster in terms of number of iterations it can take longer to calculate each iteration.
# Considering Q-learning and SARSA they have similar behaviors because they only update the Q-function. They differ in how they update those values, Q-learning (off-policy) does not depend on the policy used so in the long run it will converge to the Q-values off the optimmum policy. SARSA (on-policy) will converge to optimmum Q-values for that specific policy used. if we have infinite iterations, when the policy is not optimal Q-learning will achieve better results than SARSA.