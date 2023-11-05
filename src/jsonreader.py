from utilities.jsonRW import readJSON, readJSONById, display_agent_data, delete_agent_data_by_id

shape = "14x14"       # "5x5" or "14x14"
algorithm = "DDQN"    # "Q-Learning" or "DQN" or "DDQN"




readJSON(algorithm, shape)

#readJSONById(algorithm, shape, 1)

#display_agent_data(algorithm, shape, 2)