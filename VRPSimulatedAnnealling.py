# Author : Dominic Keehan

import random
from datetime import datetime
import numpy as np
import copy
from matplotlib import pyplot as plt
import time
import pandas as pd

random.seed(datetime.now())

def Run():
    '''
    Main Function    
    '''

    Demands = pd.read_csv('OurDemand.csv')
    Demands.set_index('Supermarket',inplace=True)
    StoreDemands=Demands['Weekday']

    LatLongs = pd.read_csv('FoodstuffLocations.csv')
    Labels = LatLongs.Supermarket
    VRPNodes=list([Labels[i] for i in range(1,len(Labels))]) # Do all nodes
    LatLongs.set_index('Supermarket',inplace=True)

    TravelTimesData = pd.read_csv('FoodstuffTravelTimes.csv')
    TravelTimesData.set_index('Unnamed: 0',inplace=True)
    TravelTimes=TravelTimesData

    plt.style.use('ggplot')
    S = RandomSolution(VRPNodes,StoreDemands,TravelTimes)
    print("Random solution: ", S)
    TotalTime=CostFunction(S,StoreDemands,TravelTimes)
    TotalTime = int((float(TotalTime)+6*60-1)/(6*60))*6*60 # Preform 6 minute ceiling
    print("Random cost: ", TotalTime)
    plt.figure()
    plt.title('Random Solution: '+str(TotalTime)+' seconds')
    PlotVRP(VRPNodes,S,LatLongs,StoreDemands)
    plt.savefig("RandomNetwork",bbox_inches='tight',dpi=1000)

    start = time.time()
    Best=Anneal(VRPNodes,StoreDemands,S,TravelTimes)
    end = time.time()

    print("Minutes to imporve: ", (end-start)/(60))
    print("Improved solution: ", Best)
    TotalTime=CostFunction(Best,StoreDemands,TravelTimes)
    TotalTime = int((float(TotalTime)+6*60-1)/(6*60))*6*60 # Preform 6 minute ceiling
    print("Improved cost: ", TotalTime)
    plt.figure()
    plt.title('Improved Solution: '+str(TotalTime)+' seconds')
    PlotVRP(VRPNodes,Best,LatLongs,StoreDemands)
    plt.savefig("ImprovedNetwork",bbox_inches='tight',dpi=1000)
    
    plt.show()

    pass


def PlotVRP(VRPNodes,Routes,LatLongs,StoreDemands):
    '''
    Plots a directed network with stores names, demands, and directions.

        Parameters:
        -----------
        VRPNodes : Array
            List of all stores used within the network.
        Routes : Array
            List of Routes used in the network. Routes are such that store 
            names visited in order. Starts at warehouse, ends at last store 
            before returning to the warehouse.
        LatLongs : DataFrame
            Store latitude and longitude indexed by store name.
        StoreDemands : Dictionairy
            Store pallet demands indexed by store name.

        Returns:
        --------
        None
    ''' 
    for Route in Routes:
        Start = Route[0] # Initialise 'from' node
        Origin = Route[0] # Get warehouse location

        color = np.random.rand(3,) # Differentiate routes by random color
        color[0]=0
        
        for R in Route:
            ConnectPoints(LatLongs['Long'][Start], LatLongs['Long'][R], LatLongs['Lat'][Start], LatLongs['Lat'][R], color)
            Start=R # Update 'from' node
        # Make route a cycle
        ConnectPoints(LatLongs['Long'][R], LatLongs['Long'][Origin], LatLongs['Lat'][R], LatLongs['Lat'][Origin], color) 

    # Plot node information
    for N in VRPNodes:
        plt.plot(LatLongs['Long'][N], LatLongs['Lat'][N], 'mo', markersize=4,alpha=0.8)
    #    plt.text(LatLongs['Long'][N], LatLongs['Lat'][N]+0.01,StoreDemands[N])
    #    plt.text(LatLongs['Long'][N]-0.015, LatLongs['Lat'][N]+0.02,N)
        
    # Labelling
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.plot(LatLongs['Long'][Origin], LatLongs['Lat'][Origin], 'ko', markersize=5) # Highlight warehouse
    #plt.ylim(-37.1, -36.6)
    #plt.xlim(174.55, 175)

    pass


def RouteDemand(Route,StoreDemands):
    '''
    Return the total demand delivered on a route

        Parameters:
        -----------
        Route : Array
            List of store names visited in order. Starts at warehouse, ends at last
            store before returning to the warehouse.
        StoreDemands : Dictionairy
            Store pallet demands indexed by store name.

        Returns:
        --------
        TotalDemand : Integer
            Total pallet deliveries on 'Route'.
    '''
    TotalDemand = 0
    # Cycle through each store, skipping the warehouse
    for i in range(1,len(Route)):
        TotalDemand = TotalDemand+StoreDemands[Route[i]]
    
    return TotalDemand


def RouteTime(Route,StoreDemands,TravelTimes):
    '''
    Return the time in seconds it takes to traverse a route and deliver its demand.
    
        Parameters:
        -----------
        Route : Array
            List of store names visited in order. Starts at warehouse, ends at last
            store before returning to the warehouse.
        StoreDemands : Dictionairy
            Store pallet demands indexed by store name.
        TravelTimes : DataFrame
            Store travelling to times, indexed by starting store

        Returns:
        --------
        TotalTime : Float
            Total delivery time of Route.
    '''
    TotalTime = 0.0

    # Cycle through each portion of the route, except the final 
    for i in range(0,len(Route)-1):
        TotalTime = TotalTime+TravelTimes[Route[i+1]][Route[i]]
    
    # Add final portion
    TotalTime = TotalTime+TravelTimes[Route[0]][Route[-1]]+RouteDemand(Route,StoreDemands)*5*60 # Add unloading time
    
    return TotalTime


def CostFunction(Routes,StoreDemands,TravelTimes):
    '''
    Objective function for simulated annealing algorithm. Returns total delivery time of all routes.
    
        Parameters:
        -----------
        Routes : Array
            List of Routes used in the network. Routes are such that store 
            names visited in order. Starts at warehouse, ends at last store 
            before returning to the warehouse.
        StoreDemands : Dictionairy
            Store pallet demands indexed by store name.
        TravelTimes : DataFrame
            Store travelling to times, indexed by starting store

        Returns:
        --------
        Cost : Float
            Total delivery time of entire set of Routes.
    '''
    Cost = 0
    for Route in Routes:
        Cost = Cost+RouteTime(Route,StoreDemands,TravelTimes)

    return Cost


def TravelToMeTime(Route,StoreIndex,TravelTimes):
    '''
    Calculates the traversal time of a portion of a route; from the previous store to the indexed store.
        
        Parameters:
        -----------
        Route : Array
            List of store names visited in order. Starts at warehouse, ends at last
            store before returning to the warehouse.
        StoreIndex : Int
            Store index with respect to Route
        TravelTimes : DataFrame
            Store travelling to times, indexed by starting store

        Returns:
        --------
        Time : Float
            Travel time from previous node to indexed node.
    '''
    Time = TravelTimes[Route[StoreIndex]][Route[StoreIndex-1]]

    return Time


def AverageTime(Route,StoreIndex,TravelTimes):
    '''
    Calculates the average travel time to and from a store.

        Parameters:
        -----------
        Route : Array
            List of store names visited in order. Starts at warehouse, ends at last
            store before returning to the warehouse.
        StoreIndex : Int
            Store index with respect to Route
        TravelTimes : DataFrame
            Store travelling to times, indexed by starting store

        Returns:
        --------
        AverageTime : Float
            Average travel time to and from indexed store.
    '''

    if StoreIndex+1 == len(Route): # Send next node to origin if appropriate
        NextIndex = 0
    else:
        NextIndex = StoreIndex+1

    AverageTime=(TravelTimes[Route[StoreIndex]][Route[StoreIndex-1]]+TravelTimes[Route[NextIndex]][Route[StoreIndex]])/2.0

    return AverageTime


def RandomSolution(Stores,StoreDemands,TravelTimes):
    '''
    Produces a random but viable VRP solution

        Parameters:
        -----------
        Stores : Array
            List of all stores used within the network.
        StoreDemands : Dictionairy
            Store pallet demands indexed by store name.
        TravelTimes : DataFrame
            Store travelling to times, indexed by starting store

        Returns:
        --------
        Routes : Array
            List of Routes used in the network. Routes are such that store 
            names visited in order. Starts at warehouse, ends at last store 
            before returning to the warehouse.
    '''

    RemainingStores = copy.deepcopy(Stores) # Avoid mutating input
    random.shuffle(RemainingStores) # Re order for random cycling

    Routes = list()
    v = 0 # Vehicle route count

    while len(RemainingStores)>0: # Cycle until all demand satisfied
        
        Routes.append(['Warehouse']) # New route
        r = 0 # Remaining stores index

        # Iterate routes until at capacity or all demand is met
        while  np.any([RouteDemand(Routes[v]+[i],StoreDemands)<12 for i in RemainingStores]) and len(RemainingStores)>0:
            
            # Check that next store is viable
            ViableDemand = RouteDemand(Routes[v]+[RemainingStores[r]],StoreDemands) <= 12
            ViableTime = RouteTime(Routes[v]+[RemainingStores[r]],StoreDemands,TravelTimes) <= 14400
            if ViableDemand and ViableTime:
                Routes[v].append(RemainingStores[r])
                RemainingStores.pop(r)
                r = 0 # Continue searching through remaining stores

            elif r < len(RemainingStores)-1:
                r = r+1 # Go to next remaining store 
            else:
                break # Begin new route if no other store fits

        v = v+1 # Next vehicle

    return Routes


def ReplaceHighestAverage(Routes,StoreDemands,TravelTimes):
    '''
    Solution transform that tries to move stores that contribute the most to the time, on average.
        
        Parameters:
        -----------
        Routes : Array
            List of Routes used in the network. Routes are such that store 
            names visited in order. Starts at warehouse, ends at last store 
            before returning to the warehouse.
        StoreDemands : Dictionairy
            Store pallet demands indexed by store name.
        TravelTimes : DataFrame
            Store travelling to times, indexed by starting store

        Returns:
        --------
            Updated list of routes in the newtork
        Notes:
        ------
        This selects the 4 stores within all the routes with the highest average input and output 
        times, and removes them from their associated routes. All the feasible insertions of these 
        four stores into the resultant routes are then computed. The resulting route which has the 
        least travel time is then accepted for each removed store, which is then inserted accordingly. 
        For this transform specifically a deterministic approach was applied as it is unlikely a 
        global optimum exists within the neighbourhood of solutions early in the algorithm’s cycle 
        with single nodes that are very far away from <n-1,n+1>. The resulting solution is then 
        checked to make sure it visits all required stores, although this should be guaranteed anyway 
        based on the initial solution input to it.

    '''
    RoutesToMut = copy.deepcopy(Routes) # Stored in case transform can't find viable solution
    random.shuffle(RoutesToMut) # Neccesary?
    
    N=4 # Number of stores to move
    StoreWeights = list([])
    
    for Route in RoutesToMut: # Cycle through routes
        for j in range(1,len(Route)): # Cycle through stores
            # Keep [Store, AverageTime]. Doesn't calulate depot as this can't be moved
            StoreWeights.append([Route[j],AverageTime(Route,j,TravelTimes)]) 
    
    LargestStores = sorted(StoreWeights, key = lambda T : T[1])[-N+1:-1] # Sort by time keeping N largest stores
    LargestStoreList = [LargestStores[i][0] for i in range(0,len(LargestStores))]

    # Cycle through routes and remove largest stores
    for L in LargestStoreList:
        for R in range(0,len(RoutesToMut)):
            if L in RoutesToMut[R]:
                RoutesToMut[R].remove(L)
                break           

    # Cycle through stores to put back in
    for L in LargestStoreList:
        LScore = list([])
        i = 0
        # Cycle through routes
        for Route in RoutesToMut:
            # Cycle through all possible store insertions
            for j in range(1,len(Route)):
                R = copy.deepcopy(Route)
                R.insert(j,L)
                Score = RouteTime(R,StoreDemands,TravelTimes)
                if Score <= 14400 and RouteDemand(R,StoreDemands) <= 12:
                    LScore.append([i,j,Score]) # Keep [Route number,insertion point,Time]
            i = i+1
        if len(LScore) != 0: # If a new solution(s) was found, keep the best one. (Deterministic)
            SmallestPScore = sorted(LScore, key = lambda P : P[2])[0] # Sort possible insertions by time
            RoutesToMut[SmallestPScore[0]].insert(SmallestPScore[1],L)

    # Check new solution still uses all the nodes
    NewStoresUsed = int(sum([len(RoutesToMut[i])-1 for i in range(0,len(RoutesToMut))]))
    OldStoresUsed = int(sum([len(Routes[i])-1 for i in range(0,len(Routes))]))
    if  NewStoresUsed<OldStoresUsed:
        
        return Routes
    else:

        return RoutesToMut

def Move(Routes,StoreDemands,Stores,TravelTimes):
    '''
    Solution transform that randomly mutates from all but the 4 smallest time to stores
        Parameters:
        -----------
        Routes : Array
            List of Routes used in the network. Routes are such that store 
            names visited in order. Starts at warehouse, ends at last store 
            before returning to the warehouse.
        StoreDemands : Dictionairy
            Store pallet demands indexed by store name.
        Stores : Array
            List of all stores used within the network.
        TravelTimes : DataFrame
            Store travelling to times, indexed by starting store

        Returns:
        --------
            Updated list of routes in the newtork  

        Notes:
        ------
        This selects the 4 stores from their associated routes with the lowest individual
        input time <n-1,n>, and removes them from the selection pool. Then from the 
        remaining stores 4 are selected at random and removed from their current routes. 
        These are then randomly inserted into other routes if the resultant route is viable. 
        Thus, if a route exists in the solution that has all of its stores linked with short 
        travel times (ie is a good route), it is unlikely to be changed, apart from as a 
        result of one of the randomly selected stores being put in. While the remaining routes 
        which are comparatively worse, are. This strikes a good balance between exploring the 
        neighbourhood and converging. The resulting solution is then checked to make sure it 
        visits all required stores, although this should be guaranteed anyway based on the initial 
        solution input to it.
  
    '''
    RoutesToMut = copy.deepcopy(Routes) # Stored in case transform can't find viable solution
    
    M=4 # Number of stores to move
    Weights = list([])
    i = 0

    # Calculate all edge travel times
    for Route in RoutesToMut:
        for j in range(0,len(Route)):
            Weights.append([Route[j],TravelToMeTime(Route,j,TravelTimes)])
            i = i+1

    LeastEdges = sorted(Weights, key = lambda W : W[1])[0:M] # Sort by travel to time, keeping M smallest nodes
    LeastEdgeList = [LeastEdges[i][0] for i in range(0,len(LeastEdges))]

    Picks = list([])
    To_Pick_From = copy.deepcopy(Stores)

    random.shuffle(To_Pick_From) # Neccesary?

    # Create random list of stores not in the best list, to mutate 
    while len(Picks)<4: 
        Picks.append(To_Pick_From.pop())
        if Picks[-1] in LeastEdgeList:
            Picks.pop(-1)

    # Cycle through routes and remove picked stores
    for R in range(0,len(RoutesToMut)):
        Node_Numbers = len(RoutesToMut[R])
        N = 0
        while N<Node_Numbers:
            for P in Picks:
                if RoutesToMut[R][N] == P:
                    RoutesToMut[R].pop(N)
                    N = N-1
                    Node_Numbers = len(RoutesToMut[R])
            N = N+1

    # Randomly put picked store back in, keep it if the route is still viable. Make this deterministic/less inefficient?
    for L in Picks:
        for R in range(0,len(RoutesToMut)):
            Insert_Point = random.randint(1,len(RoutesToMut[R]))
            RoutesToMut[R].insert(Insert_Point,L)
            if RouteDemand(RoutesToMut[R],StoreDemands) <= 14400 and RouteDemand(RoutesToMut[R],StoreDemands) <= 12:
                break
            else: 
                RoutesToMut[R].pop(Insert_Point)

    # Check new solution still uses all the nodes
    NewStoresUsed = int(sum([len(RoutesToMut[i])-1 for i in range(0,len(RoutesToMut))]))
    OldStoresUsed = int(sum([len(Routes[i])-1 for i in range(0,len(Routes))]))
    if  NewStoresUsed<OldStoresUsed:
        
        return Routes
    else:

        return RoutesToMut

def Neighbour(Routes,StoreDemands,Stores,TravelTimes):
    '''
    Function that applies solution transforms. Produces a solution in the neighborhood of previous.
        Parameters:
        -----------
        Routes : Array
            List of Routes used in the network. Routes are such that store 
            names visited in order. Starts at warehouse, ends at last store 
            before returning to the warehouse.
        StoreDemands : Dictionairy
            Store pallet demands indexed by store name.
        Stores : Array
            List of all stores used within the network.
        TravelTimes : DataFrame
            Store travelling to times, indexed by starting store

        Returns:
        --------
            Updated list of routes in the newtork
    '''
    # I had major issues with pythons pointers accessing information it shouldn't have, please escuse all the 'deepcopy'
    Out = copy.deepcopy(Routes)
    if random.uniform(0,1)<0.8: # Apply 'move' 80% of the time
        Out = Move(Routes,StoreDemands,Stores,TravelTimes)
        OutO = copy.deepcopy(Out)
        Ret = ReplaceHighestAverage(OutO,StoreDemands,TravelTimes)
 
        return Ret    
    Ret = ReplaceHighestAverage(Out,StoreDemands,TravelTimes)
 
    return Ret

def ConnectPoints(x1,x2,y1,y2,color):
    '''
    Plots a directed arrow between two cartesian coordinates.
        Parameters:
        -----------
        x1,x2,y1,y2 : Float
            Coordiantes of points
        color : Array
            RGB(A)? color value

        Returns:
        --------
            None   
    '''
    Width=0.0015
    if x1 != x2 or y1 != y2:
        plt.arrow(x1,y1,x2-x1,y2-y1,color = color,length_includes_head=True,width=Width,head_width=4*Width,head_length=6*Width,alpha=0.45)

def Anneal(VRPNodes,StoreDemands,RandomSol,TravelTimes):
    '''
    Heuristic network method that tries to find a 'good' global solution.
        Parameters:
        -----------
        VRPNodes : Array
            List of all stores used within the network.
        StoreDemands : Dictionairy
            Store pallet demands indexed by store name.
        RandomSol : Array
            Initial solution. List of Routes used in the network. Routes are such that store 
            names visited in order. Starts at warehouse, ends at last store 
            before returning to the warehouse.
        TravelTimes : DataFrame
            Store travelling to times, indexed by starting store

        Returns:
        --------
        BestS : Array
        The best solution (lowest travel time) found by the algorithm.
    '''

    # Initialise solution comparisons
    S=copy.deepcopy(RandomSol)
    BestS = copy.deepcopy(S)
    CurrentS = copy.deepcopy(S)
    CurrentC = CostFunction(CurrentS,StoreDemands,TravelTimes)
    BestC = CostFunction(BestS,StoreDemands,TravelTimes)

    a = 0.5 # Cooling factor. This signifigantly effects solving time. 0.8 seems good for all (Don't do higher than 0.85 for all).
    b = 1.05 # Grows iterations per cycle
    M0 = 5 # Iterations per cycle
    T = 10000
    Time = 0
    Max_Time = 100000 # Max out time

    print('Running SA')

    while Time<Max_Time and T>0.001:
        M = M0
        while M >= 0:
            NewS = Neighbour(CurrentS,StoreDemands,VRPNodes,TravelTimes)
            NewC = CostFunction(NewS,StoreDemands,TravelTimes)
            ChangeIN = NewC-CurrentC

            print(f'\rCurrent temperature: {T:.3f}, Explored cost (No wetlease): ${NewC*150/3600:.2f}', end='')

            if ChangeIN<0:
                CurrentS = copy.deepcopy(NewS)
                CurrentC = CostFunction(CurrentS,StoreDemands,TravelTimes)
                if NewC<BestC:
                    BestS = copy.deepcopy(NewS)
                    BestC = CostFunction(BestS,StoreDemands,TravelTimes)

            elif random.uniform(0,1)<np.exp(-ChangeIN/T): # Explore from a worse solution
                CurrentS = copy.deepcopy(NewS)
                CurrentC = CostFunction(CurrentS,StoreDemands,TravelTimes)
            M = M-1
        Time = Time+M0
        T = a*T
        M0 = b*M0

    
    return BestS  


Run()