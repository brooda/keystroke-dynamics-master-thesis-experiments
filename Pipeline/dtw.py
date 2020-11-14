import numpy as np

def createGlobalDistances(localDistances):
    m, n = localDistances.shape
    globalDistances = np.zeros((m, n))
    
    globalDistances[0, 0] = localDistances[0, 0]
    
    for i in range(1, m):
        globalDistances[i, 0] = globalDistances[i - 1, 0] + localDistances[i, 0];
    
    for i in range(1, n):
        globalDistances[0, i] = globalDistances[0, i - 1] + localDistances[0, i];

    for i in range(1, m):
        for j in range(1, n):
            globalDistances[i, j] = np.min([globalDistances[i - 1, j - 1], globalDistances[i - 1, j], 
                                            globalDistances[i, j - 1]]) + localDistances[i, j]
            
    return globalDistances


def createPath(globalDistances):
    path = []
    m, n = globalDistances.shape
    
    i = m-1
    j = n-1
    path.append((i, j))
    
    while i>0 and j>0:
        
        # B C
        # A current
        
        A = globalDistances[i, j-1]
        B = globalDistances[i-1, j-1]
        C = globalDistances[i-1, j]

        if A < B and A < C:
            # A is minimal
            j = j-1
        elif B < C:
            # B is minimal
            i = i-1
            j = j-1
        else: 
            # C is minimal
            i = i-1
        path.append((i, j))
    
    while i>0:
        i = i-1
        path.append((i, j))
    while j>0:
        j = j-1
        path.append((i, j))

    return path

def costOfPath(path, localDistances):
    cost = 0
    for coord in path:
        cost += localDistances[coord]
    
    return cost / len(path)
    
def dtw_distance(x, y):
    x = np.array(x)
    y = np.array(y)

    #print("dtw_distance",x,y)
    localDistances = np.zeros((x.size, y.size))
    
    for i in range(x.size):
        for j in range(y.size):
            localDistances[i, j] = np.sum(np.abs(x[i] - y[j]))
            
    globalDistances = createGlobalDistances(localDistances)
    path = createPath(globalDistances)
    
    return costOfPath(path, localDistances)