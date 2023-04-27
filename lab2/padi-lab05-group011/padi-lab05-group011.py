# Activity 1
import numpy as np
import numpy.random as rand
import time

def load_mdp(filename,gama):
    file = np.load(filename)
    X = file['X']
    A = file['A']
    P = file['P']
    C = file['c']
    return (tuple(X),tuple(A),tuple(P),C,gama)

# Activity 2
def noisy_policy(MDP,a, eps):
    states = MDP[0]
    actions = MDP[1]
    
    policy_matrix = np.zeros((len(states),len(actions)))
    policy_matrix = policy_matrix.astype(float)
    
    for i in range(len(states)):
        for j in range(len(actions)):
            if j == a:
                policy_matrix[i][j] = 1-eps
            else:
                policy_matrix[i][j] = eps/(len(actions)-1)
    return policy_matrix
           
# Activity 3
def evaluate_pol(MDP,policy):
    states = MDP[0]
    actions = MDP[1]
    probs = MDP[2]
    costs = MDP[3]
    gama = MDP[4]
    
    P_pi = np.zeros((len(states),len(states)))
    C_pi = np.zeros((len(states)))
    J_pi = np.zeros((len(states)))
    
    identity = np.identity(len(states))
    
    for i in range(len(actions)):
        P_pi += np.multiply(probs[i],policy[:,i,None])
    
    C_pi = np.sum(np.multiply(costs, policy), axis=1)
    C_pi = C_pi.reshape(len(C_pi),1)
    
    J_pi = np.linalg.inv(identity - gama*P_pi).dot(C_pi)
    return J_pi

# Activity 4
def value_iteration(MDP):
    states = MDP[0]
    actions = MDP[1]
    probs = MDP[2]
    costs = MDP[3]
    gama = MDP[4]
    
    error = 1
    n_iter = 0
    J = np.zeros((len(states),len(actions)))
    J_opt = np.zeros((len(states)))
    
    start = time.time()
    
    while error > 1e-8:
        n_iter+=1
        for i in range(len(actions)):
            value = probs[i].dot(J_opt)
            value = value.reshape(len(states),1)
            
            J[:,i,None] = costs[:,i,None] + gama*value
            error = np.linalg.norm(J_opt - np.min(J,axis=1))
            J_opt = np.min(J,axis=1)
            
    end = time.time()
    
    t = end - start
    print("Execution time:",  np.round(t,3), " seconds")
    print("N. iterations: ",n_iter)
    return J_opt.reshape(len(states),1)

# Activity 5
def policy_iteration(MDP):
    states = MDP[0]
    actions = MDP[1]
    probs = MDP[2]
    costs = MDP[3]
    gama = MDP[4]
    
    start = time.time()
    
    policy = np.ones((len(states),len(actions)))/len(actions)
    
    n_iter = 0
    
    values = np.zeros((len(states),len(actions)))
    while True:
        n_iter += 1
        pol_eval = evaluate_pol(MDP,policy)
        
        for a in range(len(actions)):
            values[:,a,None] = costs[:,a,None] + gama*np.dot(probs[a],pol_eval)
        values_min = np.min(values,axis=1,keepdims=True)
        
        new_pol = np.isclose(values,values_min,atol=1e-8,rtol=1e-8).astype(int)
        new_pol = new_pol/new_pol.sum(axis=1,keepdims=True)
        
        if (policy == new_pol).all():
            break
        
        policy = new_pol
        
    end = time.time()
    
    t = end - start
    print("Execution time:",  np.round(t,3), " seconds")
    print("N. iterations: ",n_iter)
    
    return policy

# Activity 6
NRUNS = 100 # Do not delete this

def simulate(MDP,policy,x0,length):
    costs = [0 for i in range(NRUNS)]
    
    states = MDP[0]
    actions = MDP[1]
    prob = MDP[2]
    cost = MDP[3]
    gama = MDP[4]
    
    curr = x0
    
    for i in range(NRUNS):
        cost_i = 0
        for j in range(length):
            a = rand.choice(len(actions),p=policy[curr,:])
            cost_i += cost[curr,a] * (gama**j)
            curr = rand.choice(len(states),p=prob[a][curr,:])
        curr = x0
        costs[i] = cost_i
    
    return np.sum(costs)/NRUNS
