from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import collections
import os
import sys
import warnings

import numpy as np

import chainerrl
from chainerrl import env
from chainerrl import spaces

from chainerrl import misc
import gym
from gym import spaces
from gym.utils import seeding
import gym.wrappers

#from copy import deepcopy, copy

import argparse
import pandas as pd
from numpy.random import *
import datetime
from os import path
import csv

import random
import math
import cmath
import hashlib
import logging

import statistics

import pickle as pickle

#import sys
sys.path.append('../')

import time
import gc

import hoo


"""
    @ Seydou
    Monte Carlo Tree Search with Variable Simulation Periods implementation for atari game.
    Tree policy: UCB1
    Default policy: Random selection
    HOOT and PW used for continuous simulation periode spaces

    Set DEPTH_MAX to reduce running time.

    Run with:
        python mcts_ata.py  --env $GAME-v0 --n_episodes $NUM_EPISODES --num_sims $MAX_SIMULATIONS --reward-scale-factor $RSF

"""

#MCTS scalar.  Larger scalar will increase exploration, smaller will increase exploitation. 
SCALAR=1/math.sqrt(2.0)
SCALAR2=10/math.sqrt(2.0)
DEPTH_MAX=300
MAX_REWARD=1
MIN_VISITS=1
C_PW = 1
T_MAX = 5
TAU_FACT = 1                # >0 if simulation budget is to be multiplied by tau

logging.basicConfig(stream=sys.stdout, level=logging.INFO) #INFO WARNING DEBUG
logger = logging.getLogger('MCTS')

class State():                      #Serve as interface between the MCTS nodes and the environment
    def __init__(self, info, state, rew=0.0, done=False, n_act=0, moves=[], 
                tau=[], Hroot=None, Hfront=None, r_act_i=[], n_act_i=[], rState=False):    #Translate the environment's state, reward, date, and 
        self.envState=state                                            #    info into the State be recorded by the node
        self.moves=moves
        self.rew=rew
        self.done=done
        self.info=info
        self.Hroots=Hroot
        self.Hfront=Hfront
        self.tau=tau
        self.n_actions= n_act
        self.r_act_i = r_act_i
        self.n_act_i = n_act_i
        self.n_visits = 0
        self.n_act_icp = n_act_i
        self.n_visits_cp = 0
        self.rootState = rState
     #   self.num_actions=state.env_space.n
        
    def next_state(self, env):              #Execute a random action in the given env and return the resulting environment state as a node State
        a = env.action_space.sample()
        if isinstance(a, np.ndarray):
            a = a.astype(np.float32)
        nextmove = [a]                              
        obs, r, done, info = env.step(nextmove)
        next = State(info, obs, self.rew+r, done)
        return next
    def expand_state(self, env):                    
        RESTOREENV(env, self.envState)          #Reset the given env to the state registered by the node
        if not self.rootState or (self.n_visits < 6 and self.n_visits < self.n_actions): # Setting \tau =1 for initial n=6 root visits and for non-root nodes 
            if not self.rootState:
                logger.debug("Expanding child %s"%(self.n_visits))
                t = 0.1
                a = env.action_space.sample()
                front=None
            else:
                logger.debug("Expanding root")
                a = self.best_ucb()
                t,front = hoo.HOOSEARCH(self.Hroots[a])
                hoo.HOOBACKUP(front,0)
                t = 0.1
        else:
            logger.debug("Expanding root")
            a = self.best_ucb()
            t,front = hoo.HOOSEARCH(self.Hroots[a])
            hoo.HOOBACKUP(front,0)
        if isinstance(a, np.ndarray):
            a = a.astype(np.float32)
        nextmove = [a] 
        #obs, r, done, info = env.step(nextmove)
        tau = int(np.floor(t))+1
        obs, r, done, info = HOOSTEP(nextmove, tau, env)
        state = CLONEENV(env)
        next = State(info, state, self.rew+r, done, env.action_space.n, self.moves+[nextmove],self.tau+[tau], Hfront=front)
            
    #    print ("Returning")
        return next
    def best_ucb(self):
        bestscore = -9999999999999.0
        bestaction=[]
        for c in range(self.n_actions):
            if self.n_act_i[c]:
                exploit = self.r_act_i[c]/self.n_act_i[c]
                logger.debug("self.n_visits_cp %s, n_act_icp %s"%(self.n_visits_cp,self.n_act_icp[c]))
                explore = math.sqrt(math.log(2*self.n_visits_cp+1)/float(self.n_act_icp[c]))  # The explore term is defined for scaled exploit 
                score=exploit+SCALAR2*explore                                # The scalar allow to scale the exploit and explore terms
            else:
                score = float("inf") # or math.inf
            if score == bestscore:                                        # Scalar is set to zero for a greedy selection
                bestaction.append(c)
            if score > bestscore:
                bestaction = [c]
                bestscore = score
        if len(bestaction)==0:
            logger.warn("OOPS: no action found, probably fatal")
        c =  random.choice(bestaction)
        self.n_act_icp[c] += 1    #Used to encourage exploration
        self.n_visits_cp += 1
        return c
    def terminal(self):
        return self.done
    def reward(self):
        return self.rew
    def __hash__(self):
        return int(hashlib.md5(str(self.moves)).hexdigest(),16)
    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False
    def __repr__(self):
        s=" move= %s, cumulated reward= %f"%(self.moves[-1],self.rew)
        return s
        
class Node():
    def __init__(self, state, parent=None):             # Create node to be attached to the tree search
        self.visits=0
        self.reward=0.0    
        self.state=state
        self.children=[]
        self.parent=parent    
    def add_child(self,child_state):          # Attach child node to the tree
        child=Node(child_state,self)
        self.children.append(child)
    def update(self,reward):                  # Update node statistics (visits, cumulated rewards)
        self.reward+=reward
        self.visits+=1
    def fully_expanded(self):                 # Verify if nodes have been created for (all possible) autorized number of actions 
        if not self.state.rootState and len(self.children)==self.state.n_actions:
            return True
        elif(len(self.children)>= C_PW * int(math.sqrt(self.visits) + 1) \
            and len(self.children)>=self.state.n_actions) or len(self.children)>=T_MAX*self.state.n_actions:    
            return True
        return False
    def showtree(self, level=0):                    # Return tree from node in the form of string
        ret = "\t"*level
        if self.state.moves: ret += repr(self.state.moves[-1])
        ret+=" "+repr("%.3f"%(self.state.envState[0]*const.HEIGHT))+" "+repr("%.3f"%(self.state.rew))+" "+repr("%d"%(self.visits))+" "+repr("%.3f"%(self.state.rew+self.reward/self.visits))+"\n"  #self.state.rew*MAX_REWARD
        for child in self.children:
            ret += child.showtree(level+1)
        return ret
    def __repr__(self):
        if not self.visits: s="New node not yet evaluated"
        rt = self
        while rt.parent != None :
            rt = rt.parent
        else: s=" Node visits= %d ; Average expected reward= %.3f"%(self.visits,(rt.state.rew+self.reward/self.visits))
        return s

def RESTOREENV(env, envState):
    env.unwrapped.close()
    env.unwrapped.restore_full_state(
                envState
            )

def CLONEENV(env):
    #env.unwrapped.close()
    state = env.unwrapped.clone_full_state()
    return state
        
def HOOSTEP(a,tau,env):
    rew = 0
    logger.debug("HOOSTEP")
    for i in range(tau): 
        obs, r, done, info = env.step(a)
        #env.render()
        rew += r
        if done: break
    return obs, rew, done, info
        
def UCTSEARCH(budget,root,env):
    logger = logging.getLogger('UCT')
    best_rew=-9999999
    worst_rew = 9999999
    j=0
    iter=0 
    tree_depth=0
    while (iter < budget and tree_depth < DEPTH_MAX) or iter < 10:      #Condition to stop simulation
        logger.debug("Tree search")
        front = TREEPOLICY(root,env)                                       
        new_depth = len(front.state.moves) - len(root.state.moves)
        # Reset env then roll till leaf node
        #env._setState(root.state.info, root.state.envState)
        RESTOREENV(env, root.state.envState)
        a = new_depth
        action_depth = 0
        while a:
            HOOSTEP(front.state.moves[-a], front.state.tau[-a],env)
        #    env.step(front.state.moves[-a])
            action_depth += front.state.tau[-a]
            a -= 1
        logger.debug("Rollout")
        reward=DEFAULTPOLICY(front.state, action_depth, env)          #(front.state,env,new_depth)
        logger.debug("Backup")
        BACKUP(front,reward-root.state.rew)
        if reward > best_rew:
            best_rew = reward
            logger.debug('Iter %s, Best reward %s', iter, best_rew)
    #    prev_depth = tree_depth
    #    if tree_depth < new_depth:
    #        tree_depth = new_depth
    #        logger.debug("Tree depth %s\n"%tree_depth) 
        iter += 1
    logger.debug("Child selection")
    logger.debug("Rewards scale from %.2f to %.2f \n"%(worst_rew-root.state.rew, best_rew-root.state.rew))
    return BESTCHILD(root,0)

def TREEPOLICY(root,env):                       # Selection step, use UCB1 for bestchild selection
    node = root
    logger.debug("Treepolicy")
    while node:                 # node.state.terminal() == False:
        if node.fully_expanded()==False:     
            return EXPAND(node,env)
        else:
            scalar = DEPTH_MAX*SCALAR     #0.1/math.sqrt(2.0)      #Used to scale exploitation and exploration terms
            node=BESTCHILD(node,SCALAR)
    return node

def EXPAND(node, env):                           # Add leaf node
    logger.debug("Expand")
    tried_children = [(c.state.moves[-1], c.state.tau[-1]) for c in node.children] 
    new_state = node.state.expand_state(env)    
    j = 0
    while (new_state.moves[-1], new_state.tau[-1]) in tried_children and j<20:    # Retry until a node corresponding to a not yet tried action is selected 
        if not node.parent:
            logger.debug("Retrying Expand n_child=%s, max_child=%s,  [%s,%s]"\
            %(len(node.children), C_PW * int(math.sqrt(node.visits) + 1), \
            new_state.moves[-1], new_state.tau[-1]))
        new_state = node.state.expand_state(env)
        j += 1
    if (new_state.moves[-1], new_state.tau[-1]) not in tried_children:
        node.add_child(new_state)                  # Attach new node to the tree
        c = node.children[-1]
    else:
        ct = [x for x in node.children if (new_state.moves[-1], new_state.tau[-1]) == (x.state.moves[-1], x.state.tau[-1])]
        c=ct[0]
    return c

def BESTCHILD(node,scalar):                     # Select best action using UCB1
    logger = logging.getLogger('UTC selection')
    logger.debug("BESTCHILD")
    bestscore = -9999999999999.0
    bestchildren=[]
    root=node
    while root.parent!=None:
        root=root.parent
    tmp_children =[]
    tmp_min_rew =[]
    min_visits = 9999
    min_rew = 9999
    
    # Pruning nodes while ignoring nodes not visited enough times 
    if node.parent==None and scalar:
        for c in node.children:
            tmp_min_rew =[]
            min_rew =9999
            for x in tmp_children:
                if x.reward/x.visits < min_rew:
                    min_rew = x.reward/x.visits
                    if(c.visits > 9): tmp_min_rew.append(x)
            if c.visits < min_visits:            # The node with min_reward is removed but the one with min_visits stays
                min_visits = c.visits
            if len(tmp_children) < 9 or c.visits < 9:
                tmp_children.append(c)
            else:
                if c.reward/c.visits >= min_rew:
                    tmp_children.append(c)
                    if c.reward/c.visits == min_rew:
                        tmp_min_rew.append(c)
                tmp_children[:] = [x for x in tmp_children if (x.reward/x.visits>=min_rew or x.visits<10)]
            if len(tmp_children) > 9 and len(tmp_min_rew):
                tmp_children.remove(random.choice(tmp_min_rew))
   #     logger.info("Selecting")
    else:
        tmp_children[:] = [x for x in node.children]
        
    for c in tmp_children:
        exploit = c.reward/c.visits               
        explore = math.sqrt(math.log(2*c.parent.visits)/float(c.visits))  # The explore term is defined for scaled exploit 
        score=exploit+scalar*explore                                # The scalar allow to scale the exploit and explore terms
        if score == bestscore:                                        # Scalar is set to zero for a greedy selection
            bestchildren.append(c)
        if score > bestscore:
            bestchildren = [c]
            bestscore = score
    if len(bestchildren)==0:
        logger.warn("OOPS: no best child found, probably fatal")
    return random.choice(bestchildren)

def DEFAULTPOLICY(state,depth,env):                   #Rollout simulation for allowed time
    logger.debug("DEFAULTPOLICY")
    t=depth
    reward = state.rew
    done = state.terminal()
 #   RESTOREENV(env, state.envState)
    while not done and t < DEPTH_MAX:
        a = env.action_space.sample()
        if isinstance(a, np.ndarray):
            a = a.astype(np.float32)
        nextmove = [a]                              
        obs, r, done, info = env.step(nextmove)
        reward += r*(0.99**t)
        t += 1
    if done: 
        env.reset()
    return reward

def BACKUP(node,reward):                        #Update visited nodes' statictic 
    logger.debug("BACKUP")
    while node!=None:
        node.visits+=1
        node.reward+=reward
        if node.parent and not node.parent.parent:                               # Root children
            c = node.state.moves[0]
            c = c[0]
            node.parent.state.n_act_i[c] += 1
            node.parent.state.r_act_i[c] += reward
            front=node.state.Hfront
            if front: 
                front.reward = node.reward/node.visits                  #Passing average reward 
        if not node.parent: 
            node.state.n_visits += 1                # Present number of runs
      #      logger.info("Backed up %s"%(node.state.n_visits))
        node=node.parent
    return

def UPDATELEGACY(node,env):
    parent = node.parent
    if parent:
        RESTOREENV(env, parent.state.envState)
        obs, r, done, info = env.step(node.state.moves[-1])
        node.state.envState = CLONEENV(env)
        node.state.rew = parent.state.rew + r
        node.state.done=done
    for c in node.children:
        UPDATELEGACY(c,env)
    return
    
start_time = time.time()    
if __name__=="__main__":
  #  logger = logging.getLogger('Main')
    parser = argparse.ArgumentParser(description='MCTS research code')
    parser.add_argument('--env', type=str, default='SpaceInvaders-v0') # Asterix Freeway BeamRider Seaquest among others
    parser.add_argument('--num_sims', action="store", type=int, default=500)
    parser.add_argument('--n_episodes', action="store", type=int, default=1)
    parser.add_argument('--notes', type=str, default='RAS')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=None)    
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    args=parser.parse_args()

    if args.seed is not None:
        misc.set_random_seed(args.seed)
        
    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)
    
    def reward_filter(r):
        return r * args.reward_scale_factor

    def make_env():
        env = gym.make(args.env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, OUT_DIR)
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        misc.env_modifiers.make_reward_filtered(env, reward_filter)
        if args.render:
            misc.env_modifiers.make_rendered(env)
        def __exit__(self, *args):
            pass
        env.__exit__ = __exit__
        return env

    n_episodes = args.n_episodes
    n_sims = args.num_sims
    
    env = make_env()            #Evaluation environment
    max_episode_len = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_size = np.asarray(env.observation_space.shape).prod()
    action_space = env.action_space
    env.reset()
    
    logger.info('Maximum episode length %s, number of actions %s'%(max_episode_len,action_space.n))
    logger.info('Reward scale %s, Tmax %s, Max_depth %s, Number of simulation %s'%(args.reward_scale_factor, T_MAX, DEPTH_MAX, n_sims))
    
    MCTS_env = make_env()      #Simulation environment
    MCTS_env.reset()
    
    scores = []
    prev_time = start_time
    for i in range(n_episodes):
        env.reset()
        state = CLONEENV(env)
   #     print (state)
        done = False
        test_r = 0
        t = 0
        tau = 1
        
       # Create root node for the HOOT of expension node
        Hroot = hoo.HooNode([0, T_MAX])
        new_Hroots = [Hroot for c in range(action_space.n)]
        n_i = [0 for c in range(action_space.n)]
        r_i = [0 for c in range(action_space.n)]
        current_node = Node(State(None,state,n_act=action_space.n,Hroot=new_Hroots, n_act_i=n_i, r_act_i=r_i, rState=True))
        
        while not (done or t == max_episode_len):
            rew = 0
            if t==0:
                current_node = UCTSEARCH(n_sims,current_node,MCTS_env)
                a = current_node.state.moves[-1]
                tau = current_node.state.tau[-1]
                logger.info(" Selected a and tau [%s, %s]"%(a[0],tau))
                prev_root = current_node.parent
                for c in prev_root.children:
                    logger.debug(" [%s %s] visit %s, value %s"%(c.state.moves[-1], c.state.tau[-1], c.visits, c.reward/c.visits))
                lead_node = current_node
 
            Hroot = hoo.HooNode([0, T_MAX])
            new_Hroots = [Hroot for c in range(action_space.n)] 
            n_i = [0 for c in range(action_space.n)]
            r_i = [0 for c in range(action_space.n)]
            #    current_node = Node(State(None,state,n_act=action_space.n,Hroot=new_Hroots, n_act_i=n_i, r_act_i=r_i, rState=True))
            if tau == 1:
                current_node = lead_node           # For tau=1 after initial selections
                
                obs, r, done, info = env.step(a)
                #    env.render()
                rew += r
                t += 1
                state = CLONEENV(env)
                
                current_node.state.envState = state
                current_node.state.rew = test_r*args.reward_scale_factor
           #     current_node.state.done = done
                current_node.parent = None
                current_node.state.Hroots = new_Hroots
                current_node.state.n_act_i = n_i
                current_node.state.r_act_i = r_i
                current_node.state.n_act_icp = n_i
                current_node.state.Hfront = None
                current_node.state.rootState = True
                tmp_children =[]
                min_rew = 0
                for c in current_node.children:
                    # logger.info("Children before")
                    if len(tmp_children) < 6:
                        tmp_children.append(c)
                    else:
                        min_rew = [x.reward for x in tmp_children if x.reward < min_rew]
                        if c.reward > min_rew:
                            tmp_children.append(c)
                        tmp_children[:] = [x for x in tmp_children if x.reward!=min_rew]
                current_node.children[:] = [x for x in tmp_children]
                logger.debug("Children after")
                for c in current_node.children:
                    logger.debug(" [%s %s] reward %s, visit %s"%(c.state.moves[-1], c.state.tau[-1], c.reward, c.visits))
                        
                UPDATELEGACY(current_node,MCTS_env)
                lead_node = UCTSEARCH(n_sims,current_node,MCTS_env)
                
            else:
                current_node = Node(State(None,state,n_act=action_space.n,Hroot=new_Hroots, n_act_i=n_i, r_act_i=r_i, rState=True))

                for j in range(tau): 
                    obs, r, done, info = env.step(a)
                #    env.render()
                    rew += r
                    t += 1
                    if done: break
                    
                    state = CLONEENV(env) 
                    RESTOREENV(MCTS_env, state)                    
                    rew2 = 0
                    done2 = False
                    for _ in range(tau-j-1):                                #Update estimated state after each step
                        obs2, r2, done2, info = MCTS_env.step(a)
                        rew2 += r2
                    current_node.state.envState = CLONEENV(MCTS_env)
                    current_node.state.rew = test_r*args.reward_scale_factor + rew + rew2
               #     current_node.state.done=done2
                    UPDATELEGACY(current_node,MCTS_env)
                    lead_node = UCTSEARCH(n_sims,current_node,MCTS_env)     #In case tau>1 current_node is reussed
               #     logger.info(" Root children %s"%len(current_node.children))
            
            prev_root = current_node
            for c in prev_root.children:
                logger.debug(" [%s %s] visit %s, value %s"%(c.state.moves[-1], c.state.tau[-1], c.visits, c.reward/c.visits))
                
            a = lead_node.state.moves[-1]
            tau = lead_node.state.tau[-1]
            b = np.float32(a)
            test_r += rew/args.reward_scale_factor          # Un-scaled Reward
            
            if t%(T_MAX*20)<=T_MAX:                         # To follow progress
                logger.info(" \n Episode %s, Step %s, Reward %s. \n Selected a and tau [%s, %s] \n"%(i, t, test_r, a[0], tau))
                
        scores.append(float(test_r))
        mean = np.mean(scores)
        median = np.median(scores)
        logger.info('test episode %s, R %s, av. %r', i, test_r, mean)
        print("--- Episode finished in %s seconds --- \n" % (time.time() - prev_time))
        prev_time = time.time()
    if n_episodes >= 2:
        stdev = np.std(scores)
    else:
        stdev = 0.

    #return mean, median, stdev
    logger.info('Test episodes mean, median, stdev: %s, %s, %s', mean, median, stdev)
    
print("--- %s seconds ---" % (time.time() - start_time)) 
