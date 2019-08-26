# Monte Carlo Tree Search with Variable Simulation Periods

VSP-MCTS implementation for for Continuously Running Tasks.

    - Tree policy: UCB1
    - Default policy: Random selection
    - Expesion: HOO with PW
 
Run with python 2.7 and chainerrl

  - Classical control tasks with continuous action space (Pendulum, MountainCarContinuous):
      - python mcts_cont.py  --env $ENV-v0 --n_episodes $NUM_EPISODES --num_sims $MAX_SIMULATIONS
      - default settings (ENV = Pendulum, NUM_EPISODES=25, MAX_SIMULATIONS = 100)

  - Atari games
      - python mcts_ata.py  --env $GAME-v0 --n_episodes $NUM_EPISODES --num_sims $MAX_SIMULATIONS --reward-scale-factor $RSF
      - default settings (ENV = SpaceInvaders, NUM_EPISODES=1, MAX_SIMULATIONS = 500, RSF=1e-2) 
