#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import argparse
import os
from os import path
import csv

from chainerrl import misc
import gym
from gym import spaces
from gym.utils import seeding
import gym.wrappers

import numpy as np
import random
import math
import cmath
import logging

import sys
sys.path.append('../')

import time
import datetime
import gc

import hoo

"""
@ Seydou
Monte Carlo Tree Search implementation for discrete action space.
Expended code from https://github.com/haroldsultan/MCTS .
Tree policy: UCB1
Default policy: Random selection

Run with:
    python mcts_dam_v2.py --num_sims BUDGET --levels NUM_ITERATION

Notes
    Changing pendulum state from [th, thdot] to [x,y,thdot]
    
    
To do
    Turn MCTS as a class to which hyper parameter can be passed to. 
    Merge codes to run Pendulum and Cont_MC with same program
    Remove all redundancy
    
    
"""

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=1/math.sqrt(2.0)
DEPTH_MAX=50
MAX_REWARD=1
C_PW = 1
T_MAX = 10

logging.basicConfig(stream=sys.stdout, level=logging.INFO) #WARNING DEBUG
logger = logging.getLogger('MCTS')

class State():                      #Serve as interface between the MCTS nodes and the environment
    MOVES=[0, 1, 2]            
    num_moves=len(MOVES)
    def __init__(self, info, state, rew=0.0, done=False, moves=[], tau=[], Hroot=None, Hfront=None):    
    #Translate the environment's state, reward and other info, along with hoo trees to be recorded into the node
        self.envState=state                                            
        self.moves=moves
        self.rew=rew
        self.done=done
        self.info=info
        self.Hroot=Hroot
        self.Hfront=Hfront
        self.tau=tau
    def next_state(self, env, d):                               # apply a random action to the given env \
        a = np.random.uniform(low=env.min_action, high=env.max_action)                         
        if isinstance(a, np.ndarray):
            a = a.astype(np.float32)
        nextmove = [a]                                                   
        obs, r, done, info = env.step(nextmove)
        st = env.state
        next = State(info, st, self.rew+r*0.99**d, done)     #Discounted reward during rollout
        return next
    def expand_state(self, env):                        
        env._setState(self.info, self.envState)      # Reset envState to the state registered by this node
        a, t, front = hoo.HOOSEARCH(self.Hroot, n_dims=True)      # Sample (a,t) from selected HOOT node (front)
        
        if isinstance(a, np.ndarray):
            nextmove = a.astype(np.float32)
        else: nextmove = [a]
        tau = int(np.floor(t))+1
        obs, r, done, info = hoostep(nextmove, tau, env)        
      #  obs, r, done, info = env.step(nextmove)
        st = env.state
        hoo.HOOBACKUP(front,r)
        new_Hroot = hoo.HooNode([-env.max_action, env.max_action, 0, T_MAX])  # Create Hroot for the HOOT of expension node
        next = State(info, st, self.rew+r, done, self.moves+[nextmove], self.tau+[tau], new_Hroot, front)
        return next
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
    def add_child(self,child_state):                    # Attach child node to the tree
        child=Node(child_state,self)
        self.children.append(child)
    def update(self,reward):                            # Update node statistics (visits, cumulated rewards)
        self.reward+=reward
        self.visits+=1
    def fully_expanded(self):                           # Verify if nodes have been created for (all possible) autorized number of actions 
        if len(self.children)== C_PW * int(math.sqrt(self.visits) + 1):    # len(self.children)==self.state.num_moves:
            return True
        return False
    def showtree(self, level=0):                    # Return tree from node in the form of string
        ret = "\t"*level
        if self.state.moves: ret += repr(self.state.moves[-1])
        ret+=" "+repr("%.3f"%(self.state.envState[0]*const.HEIGHT))+" "+repr("%.3f"%(self.state.rew))+" "+repr("%d"%(self.visits))+" "+repr("%.3f"%(self.state.rew+self.reward/self.visits))+"\n"  #self.state.rew
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
   
class PendulumEnv():        # Stand-in model for Gym Pendulum-v0, taylored for MCTS use 

    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.min_action = -self.max_torque
        self.max_action = self.max_torque
        self.dt=.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        a,b,thdot = self.state              # th := theta
        th = cmath.phase(complex(a,b))
        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

     #   u = np.clip(u, -self.max_torque, self.max_torque)[0]
        u=u[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        if newthdot < -self.max_speed: 
            newthdot = -self.max_speed 
        if newthdot > self.max_speed: 
            newthdot = self.max_speed 
    #    newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([np.cos(newth), np.sin(newth), newthdot])
        return self._get_obs(), -costs, False, self.last_u            # Added last_u as info, default is {}

    def _reset(self):
        high = np.array([np.pi, 1])
        theta, thetadot = self.np_random.uniform(low=-high, high=high)
        self.state = np.array([np.cos(theta), np.sin(theta), thetadot])
        self.last_u = None
        return self.state

    def _setState(self, last_u, obs_state):
        self.state = obs_state
        self.last_u = last_u
        return 
        
    def _get_obs(self):
        return self.state

class Continuous_MountainCarEnv():  # Stand-in model for Gym Continuous_MountainCarEnv, taylored for MCTS use

    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,))
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)

        reward = 0 # + (position**2)*0.1
        if done:
            reward = 100.0
        reward -= (action[0]**2)*0.1

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _setState(self, last_u, obs_state):
        self.state = obs_state
        self.last_u = last_u
        return self.state

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

def hoostep(a,tau,env):     # Apply selected action for the corresponding 
    rew = 0
    for i in range(tau): 
        obs, r, done, info = env.step(a)
        rew += r
        if done: break
    return obs, rew, done, info
        
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class MCTS(object):
    def __init__(self, env, num_sims=100, depth_max=DEPTH_MAX):
        self.env = env
        self.num_sims = num_sims
        self.scalar = 1/np.sqrt(2.0)
        self.depth_max = depth_max
        self.max_reward = 1
        
    def uctsearch(self, root, budget=None):
        logger = logging.getLogger('UCT')
        if budget is None: 
            budget = self.num_sims
        best_rew=-99999999
        worst_rew = 99999999
        j=0
        iter=0 
        tree_depth=0
        while iter < budget and tree_depth < self.depth_max:      # Condition to stop simulation
            if iter%1000==999:
                logger.debug("\t Simulation: %d"%iter)
                logger.debug(root)
                for i,c in enumerate(root.children):
                    logger.debug(" %s, %s"%(i,c))
            
            logger.debug("Tree search")        
            front = self.treepolicy(root)                         # Selection and expansion                          
            new_depth = len(front.state.moves) - len(root.state.moves)

            a = new_depth                           # Number of Nodes between root and selected leaf node 
            action_depth = 0            
            while a:
                action_depth += front.state.tau[-a] # Actual number of steps between root and selected node
                a -= 1
                
            logger.debug("Rollout")
            reward = self.defaultpolicy(front.state, action_depth)          #Rollout front leaf node (front)
            if reward > best_rew:
                best_rew = reward
            if reward < worst_rew:
                worst_rew = reward

            logger.debug("Backup")
            self.backup(front,reward-root.state.rew)
            prev_depth = tree_depth
            if tree_depth < new_depth:
                tree_depth = new_depth
                logger.debug("Tree depth %s\n"%tree_depth) 
            iter += 1
        logger.debug("Child selection")
        logger.debug("Rewards scale from %.2f to %.2f \n"%(worst_rew-root.state.rew, best_rew-root.state.rew))
        return self.bestchild(root,0)

    def treepolicy(self,root):                       # Selection step, use UCB1 for bestchild selection
        node = root
        while node.state.terminal()==False:
            if node.fully_expanded()==False:     
                return self.expand(node)
            else:
                scalar = self.depth_max*self.scalar           #Used to scale exploitation and exploration terms
                node=self.bestchild(node,scalar)
        return node

    def expand(self,node):                           # Add leaf node
        tried_children=[c.state.moves[-1] for c in node.children]
        new_state=node.state.expand_state(self.env)
        while new_state.moves[-1] in tried_children:        # Retry until a not yet tried action is selected 
            new_state=node.state.expand_state(self.env)
        node.add_child(new_state)                           # Attach new node to the tree
        return node.children[-1]

    def bestchild(self,node,scalar):                 # Select best action using UCB1
        logger = logging.getLogger('UTC selection')
        bestscore = -9999999999999.0
        bestchildren=[]
        root=node
        while root.parent!=None:
            root=root.parent
        for c in node.children:
            exploit = c.reward/c.visits              
            logger.debug("reward %d, visit %d, n %d"%(c.reward, c.visits, node.visits))
            explore = math.sqrt(math.log(2*root.visits)/float(c.visits)) 
            score=exploit+scalar*explore                   # UCB1 score             
            if score == bestscore:                                       
                bestchildren.append(c)
            if score > bestscore:
                bestchildren = [c]
                bestscore = score
        if len(bestchildren)==0:
            logger.warn("OOPS: no best child found, probably fatal")
        return random.choice(bestchildren)

    def defaultpolicy(self,state,depth):                   #Rollout simulation for allowed time
        t=0
        self.env._setState(state.info, state.envState)               # Set envState to the leaf node's state
        while state.terminal()==False and depth+t < self.depth_max:  #Stop rollout if terminal state or after a max. depth given by DEPTH_MAX, is reached.
            state=state.next_state(self.env,t)                       
            t += 1
        return state.reward()

    def backup(self,node,reward):                        #Update visited nodes' statictic 
        while node!=None:
            node.visits+=1
            node.reward+=reward
            front=node.state.Hfront
            if front != None: front.reward=front.visits*node.reward/node.visits
            node=node.parent
        return

start_time = time.time()
if __name__=="__main__":
    logger = logging.getLogger('Main')
    parser = argparse.ArgumentParser(description='MCTS research code')
    parser.add_argument('--env', type=str, default='Pendulum-v0')           #'Pendulum-v0 or MountainCarContinuous-v0
    parser.add_argument('--num_sims', action="store", default=200, type=int)
    parser.add_argument('--n_episodes', action="store", default=1, type=int)
    parser.add_argument('--notes', type=str, default='RAS')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=None)    
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--depth_max', type=int, default=50)
    args=parser.parse_args()
    
 #   OUT_DIR = './tmp/%d' %(time.time())         #Create subfolder in output directory
 #   if not os.path.exists(OUT_DIR):
 #       os.makedirs(OUT_DIR)
    
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

    env = make_env()     #Evaluation environment        
    max_episode_len = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
  #  print(max_episode_len)
    obs_size = np.asarray(env.observation_space.shape).prod()
    action_space = env.action_space
    
    if args.env == 'Pendulum-v0':                  #Simulation environment
        MCTS_env = PendulumEnv()                
        depth_max = 50
    else:
        MCTS_env = Continuous_MountainCarEnv()   
        depth_max = 500
    MCTS_env._reset()
    max_action = MCTS_env.max_action
            
    n_runs = args.n_episodes
    n_sims = args.num_sims
  #  depth_max = args.depth_max
    
    mcts_search = MCTS(MCTS_env, n_sims, depth_max)
    
    scores = []
    for i in range(n_runs):
        state = np.array(env.reset())
        done = False
        test_r = 0
        t = 0
        tau = 1
        while not (done or t == max_episode_len):
            # Create root node for the HOOT of expension node
            new_Hroot = hoo.HooNode([-max_action, max_action, 0, T_MAX])   
            current_node = Node(State(None,state,Hroot=new_Hroot))           #Set root node to initial state
   #         print (current_node.state.envState)
   #         print (i)
       #     if not TAU_FACT: 
       #         tau = 1
            current_node = mcts_search.uctsearch(current_node, tau*n_sims)
            if t%100<=5:
                logger.debug(" Eval episode %s, step %s:"%(i,t))
                for j,c in enumerate(current_node.parent.children):
                    logger.debug(" Child %s, %s"%(j,c))
            a = current_node.state.moves[-1]
            tau = current_node.state.tau[-1]
            b = np.float32(a)
            #   logger.info(" Step %s, \t a: %.3f, \t and t: %s"%(t,b,tau))
            state, r, done, info = hoostep(a, tau, env) # env.step(a)
            test_r += r/args.reward_scale_factor
            t += tau           # t += 1
            
        # As mixing float and numpy float causes errors in statistics
        # functions, here every score is cast to float.
        scores.append(float(test_r))
        logger.info('test episode: %s, R: %s', i, test_r)
    mean = np.mean(scores)
    median = np.median(scores)
    if n_runs >= 2:
        stdev = np.std(scores)
    else:
        stdev = 0.

    #return mean, median, stdev
    logger.info('Test episodes mean, median, stdev: %s, %s, %s', mean, median, stdev)
    
print("--- %s seconds ---" % (time.time() - start_time))                
