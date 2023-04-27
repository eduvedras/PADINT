#Acitivty 1
import numpy as np

def load_pomdp(filename,gama):
    file = np.load(filename)
    X = file['X']
    A = file['A']
    Z = file['Z']
    P = file['P']
    O = file['O']
    c = file['c']
    return (tuple(X),tuple(A),tuple(Z),tuple(P),tuple(O),c,gama)

#Activity 2
def gen_trajectory(pomdp, x0, n):
    A = pomdp[1]
    P = pomdp[3]
    O = pomdp[4]
    
    actions = []
    states = []
    observations = []
    states.append(x0)
    for i in range(n):
        rand_action = np.random.randint(len(A))
        actions.append(rand_action)
        next_s = np.random.choice(len(P[rand_action][states[i],:]), p = P[rand_action][states[i], :])
        states.append(next_s)
        obs = np.random.choice(len(O[rand_action][next_s,:]), p = O[rand_action][next_s, :])
        observations.append(obs)
    
    return (np.array(states), np.array(actions), np.array(observations))

# Activity 3
def belief_update(pomdp, belief, action, observation):
    P = pomdp[3]
    O = pomdp[4]
    
    new_belief = belief.dot(P[action]).dot(np.diag(O[action][:,observation]))
    new_belief = new_belief / np.sum(new_belief)
    
    return new_belief

def sample_beliefs(pomdp, n):
    X = pomdp[0]
    
    rand_state = np.random.choice(len(X))
    t_states,t_actions,t_observations = gen_trajectory(pomdp, rand_state, n)
    
    initial_belief = np.ones((1,len(X)))
    initial_belief = initial_belief / np.sum(initial_belief)
    
    res = [initial_belief]
    for i in range(n):
        new_belief = belief_update(pomdp, res[-1], t_actions[i], t_observations[i])
        if all([np.linalg.norm(new_belief - belief) >= 1e-3 for belief in res if new_belief is not belief]):
            res.append(new_belief)
    return tuple(res)


# Activity 4
def solve_mdp(pomdp):
    X = pomdp[0]
    A = pomdp[1]
    P = pomdp[3]
    c = pomdp[5]
    gama = pomdp[6]
    
    error = 1

    j_optimal = np.zeros(len(X))
    j = np.zeros((len(X),len(A)))
    
    while(error > 1e-8):
        for i in range(len(A)):
            v = np.dot(P[i],j_optimal)
            v = v.reshape(len(X),1)
            j[:,i,None] = c[:,i,None] + v * gama
            error = np.linalg.norm(j_optimal - np.min(j,axis=1))
            j_optimal = np.min(j,axis=1)
    
    return np.array(j)

# Activity 5
def get_heuristic_action(belief,sol, h):
    a = 0
    pol = np.zeros((len(sol),len(sol)))
    
    for i in range(len(sol)):
        minimum = np.min(sol[i])
        for j in range(len(sol[0])):
            if np.isclose(sol[i,j],minimum):
                sol[i,j] = 1
        sol[i] = sol[i] / np.sum(sol[i])
    
    if h == 'mls':
        s = np.random.choice(np.flatnonzero(belief == belief.max()))
        a = np.random.choice(np.flatnonzero(sol[s] == sol[s].max()))
    if h == 'av':
        var = belief.dot(pol)
        s = np.random.choice(np.flatnonzero(var == var.max()))
        a = np.random.choice(np.flatnonzero(sol[s] == sol[s].max()))
    if h == 'q-mdp':
        var = belief.dot(sol)
        s = np.random.choice(np.flatnonzero(var == var.max()))
        a = np.random.choice(np.flatnonzero(sol[s] == sol[s].max()))
        
    return a
             
# Activity 6
def solve_fib(pomdp):
    X = pomdp[0]
    A = pomdp[1]
    Z = pomdp[2]
    P = pomdp[3]
    O = pomdp[4]
    c = pomdp[5]
    gama = pomdp[6]
    
    error = 1

    fib = np.zeros((len(X),len(A)))
    J = np.zeros((len(X),len(A)))

    while(error > 1e-1):
        sum = np.zeros(len(X))
        for i in range(len(Z)):
            for j in range(len(X)):
                v = np.dot(P[j],O[i])
                v = np.dot(v,fib)
               
            sum = sum + np.min(v,axis=1)
            
        J[:,i,None] = c[:,i,None] + sum * gama
       
        error = np.linalg.norm(fib - np.min(J,axis=1))
        print(error)
        fib = np.min(J,axis=1)
    
    return np.array(J)

# We can conclude that FIB provides, in general, a better approximation to J* than Q-MDP since is able to better accommodate  the impact of partial observability in the decision process of the agent. 
