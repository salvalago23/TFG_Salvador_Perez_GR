from envs.createEnvs import createCSVEnv
import pygame

first_write = False
writing_enabled = False
num_executions = 1000

shape = "14x14"
render = True

if shape == "5x5":
    csv_name  = "history5x5"
elif shape == "14x14":
    csv_name  = "history14x14"


env = createCSVEnv(shape, render=render)

with open(f"../data/csv/" + csv_name + ".csv", 'a') as f:
    if first_write and writing_enabled:
        f.write(f"step,y,x,action,next_y,next_x,reward,done\n")
    
    for i in range(num_executions):
        env.unwrapped.randomize_start_pos()
        obs, _ = env.reset()
        if render: env.render()

        t = 0
        done = False
        while not done:
            if render: pygame.event.get()
            action = env.action_space.sample()  # Random action selection
            prev_state = [obs[0], obs[1], action]
            obs, rew, done, _, _ = env.step(action)

            if writing_enabled:
                f.write(f"{t},{prev_state[0]},{prev_state[1]},{prev_state[2]},{obs[0]},{obs[1]},{rew},{done}\n")
        
            t += 1
            if render:
                env.render()
                pygame.time.wait(50)
            
        print("Agente", i+1, "terminado en", t, "pasos")

if render: env.close()