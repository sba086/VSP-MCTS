import numpy as np
from numpy.random import *
import random
import math
import cmath

class HooNode():
    def __init__(self, range, h=0, parent=None):             # Create node to be attached to the tree search
        self.visits=0
        self.reward=0.0    
        self.range=range
        self.Bvalue=np.inf
        self.h=h
        self.children=[]
        self.parent=parent    
    def add_child(self,child_state):                    # Attach child node to the tree
        child=HooNode(child_state,self.h+1,self)
        self.children.append(child)
    def update(self,reward):                            # Update node statistics (visits, cumulated rewards)
        self.reward+=reward
        self.visits+=1
    def bifurc(self, n_dims):                                   # Add two leafs to node
        a=self.range[0]
        b=self.range[1]
        c = (a+b)/2
        if n_dims:                                      # n_dims if need to bifurc both action and time spaces
            x=self.range[2]
            y=self.range[3]
            if int(np.floor(x)) - int(np.floor(y)):
                z = (x+y)/2
                self.add_child([a, c, x, z])                          # Add child range as state node
                self.add_child([a, c, z, y])
                self.add_child([c, b, x, z])
                self.add_child([c, b, z, y])
            else:
                self.add_child([a, c, x, y])
                self.add_child([c, b, x, y])
        elif int(np.floor(a)) - int(np.floor(b)):        # Skip bifurc if both childs result in same tau
            self.add_child([a, c])                      
            self.add_child([c, b])
    def showtree(self, level=0):                    # Return tree from node in the form of string
        ret = "\t"*level
     #   if self.state.moves: ret += repr(self.state.moves[-1])
        if self.visits:
            ret+= repr(self.h)+repr(" %d, %.3f"%(self.visits, self.reward/self.visits))+repr(" %.3f"%(self.Bvalue))+"\n" 
        else: ret+= repr("%d, %.3f"%(self.visits, self.reward))+repr(" %.3f"%(self.Bvalue))+"\n" 
        for child in self.children:
            ret += child.showtree(level+1)
        return ret
    def __repr__(self):
        if not self.visits: s="New node not yet evaluated"
        else: s=" Node visits= %d ; Average expected reward= %.3f"%(self.visits,self.reward*MAX_REWARD/self.visits)
        return s

def HOOSEARCH(root, n_dims=False):
    B_UPDATE(root,root.visits)
    front=HOOPOLICY(root)
    front.bifurc(n_dims)
    a = np.random.uniform(low=front.range[0], high=front.range[1])
    if not n_dims:
        t = a
        return t, front
    else:
        t = np.random.uniform(low=front.range[2], high=front.range[3])
        return a, t, front

def HOOPOLICY(root):                       # Selection step, use UCB1 for bestchild selection
    node = root
    while node.children:
         node=BESTHOOCHILD(node)
    return node

def BESTHOOCHILD(node):                              # Select child with min Bvalue
    bestchildren = []
    bestscore = -np.inf
    for c in node.children:
        if c.Bvalue == bestscore:                                        # Scalar is set to zero for a greedy selection
            bestchildren.append(c)
        if c.Bvalue > bestscore:                                        # Scalar is set to zero for a greedy selection
            bestchildren = [c]
            bestscore = c.Bvalue
    if len(bestchildren)==0:
        logger.warn("OOPS: no best child found, probably fatal")
    return random.choice(bestchildren)


def HOOBACKUP(node,reward):                        #Update visited nodes' statictic 
    while node!=None:
        node.visits+=1
        node.reward+=reward
        node=node.parent
    return

def B_UPDATE(node,N):
    h=node.h
    if node.visits:
        for c in node.children:
            if c.visits: B_UPDATE(c,N)                   # Recursive loop to update all tree from leafs to root
        child_Bmax =0                                    # child_Bmax to be updated to max(child.Bvalue) 
        cumu_reward = node.reward                        # Single average reward
        for x in node.children: 
            cumu_reward += x.reward * x.visits           # Update node's total reward as sum of children's reward added to av.rew.
            if x.Bvalue > child_Bmax: child_Bmax = x.Bvalue
        node.reward = cumu_reward/node.visits
        Uvalue = node.reward + math.sqrt(2*math.log(N)/node.visits) + 1*0.5**h   # Compute Uvalue
        if Uvalue < child_Bmax: node.Bvalue = Uvalue     # Update Bvalue
        else: node.Bvalue = child_Bmax
    return

    