# VALUE ITERATION ALGORITHM

## AIM
The aim of the program is to find the best way for an agent to navigate a grid environment by using the Value Iteration algorithm.

## PROBLEM STATEMENT
The goal is to determine the optimal policy for an agent navigating through a grid environment, where each state represents a grid cell, and actions move the agent in one of four directions (up, down, left, or right). The task is to maximize the expected reward, leading the agent to the goal state while avoiding obstacles.

## VALUE ITERATION ALGORITHM
# Step 1:
Set the value of each state to 0 (initial guess).

# Step 2:
Look at all the actions you can take from that state (like moving up, down, left, or right).

# Step 3:
Calculate the expected value of each action (i.e., how good that action is based on its possible results).

# Step 4:
Pick the action that gives the highest value and update the value of the state with that number.

# Step 5:
Keep updating the values for all states until the difference between the old and new values is very small

# Step 6:
Once the values have stabilized, go through each state again and pick the action that leads to the highest value. This gives you the optimal action (policy) for each state

## VALUE ITERATION FUNCTION
### Name:Daniel c
### Register Number:212223240023
```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward+gamma*V[next_state]*(not done))
        if np.max(np.abs(V-np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi= lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return V,pi
```


## OUTPUT:
<img width="717" height="193" alt="image" src="https://github.com/user-attachments/assets/6b40ae69-2941-49b3-9753-44e564065513" />

<img width="787" height="62" alt="image" src="https://github.com/user-attachments/assets/92d30580-583d-4815-a673-f6235302f960" />

<img width="677" height="180" alt="image" src="https://github.com/user-attachments/assets/896dd7ae-9a7b-4c51-b620-9627618f6ed5" />

## RESULT:
Thus the program to find the optimal policy using value iteration is implemented successfully


