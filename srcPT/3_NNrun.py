from envs.createEnvs import createNNEnv
#import pygame

env = createNNEnv()

with open(f"../data/csv/historyNN.csv", 'a') as f:
    f.write(f"step,y,x,action,next_y,next_x,reward,done\n")
    for i in range(1):
        obs, _ = env.reset()
        #env.render()

        t = 0
        done = False
        while not done:
            #pygame.event.get()
            action = env.action_space.sample()  # Random action selection
            prev_state = [obs[0], obs[1], action]
            obs, rew, done, _, _ = env.step(action)

            f.write(f"{t},{prev_state[0]},{prev_state[1]},{prev_state[2]},{obs[0]},{obs[1]},{rew},{done}\n")
            
            t += 1
            #env.render()
            #pygame.time.wait(20)
        print("Agente", i+1, "terminado en", t, "pasos")

#env.close()