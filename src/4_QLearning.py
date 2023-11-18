from classes.QLearningAgentClass import QLearningAgent
from utilities.plots import create_grids
from utilities.jsonRW import writeJSON

#Hyperparameters
show_stats = True      # show stats
export_to_JSON = False  # write JSON file
render = True           # render the results after training

NUM_AGENTS = 2

SHAPE = "5x5"           # shape of the grid environment

"""if SHAPE == "5x5":
    EPISODES_PER_AGENT = 3000
    MAX_STEPS_PER_EPISODE = 100
elif SHAPE == "14x14":
    EPISODES_PER_AGENT = 6000
    MAX_STEPS_PER_EPISODE = 200"""

episodes = [3000]
steps = [100,150]

EPS_START = 1.0
#EPS_DECAY = EPS_START/(EPISODES_PER_AGENT/2)  # reduce the exploration over time
EPS_END = 0.1

DISCOUNT_FACTOR = 0.95
LR = 0.02



for n_episode in episodes:
    EPISODES_PER_AGENT = n_episode
    EPS_DECAY = EPS_START/(EPISODES_PER_AGENT/2)

    for n_steps in steps:
        MAX_STEPS_PER_EPISODE = n_steps

        print("Starting training with ", EPISODES_PER_AGENT, "episodes and ", MAX_STEPS_PER_EPISODE, "steps per episode")

        #CREATING AND TRAINING THE AGENTS
        agents_arr = []

        print("Starting training of", NUM_AGENTS, "Q-learning agents")

        for i in range(NUM_AGENTS):
            print("Agent", i+1, "/", NUM_AGENTS)
            agent = QLearningAgent(i, SHAPE, EPISODES_PER_AGENT, MAX_STEPS_PER_EPISODE, LR, EPS_START, EPS_DECAY, EPS_END, DISCOUNT_FACTOR)
            
            agent.train()
            agents_arr.append(agent)


        for agent in agents_arr:
            value_grid, policy_grid, string_policy_grid = create_grids(agent.env, agent=agent)
            start_pos = agent.env.unwrapped.start_pos

            if export_to_JSON:
                writeJSON(agent.algorithm, agent.n_episodes, agent.max_steps, agent.shape, start_pos, value_grid, policy_grid, string_policy_grid)
