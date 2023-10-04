import numpy as np
import tensorflow as tf

def check_done(obs):
    if obs[0][0] == 0 and obs[0][1] == 4:
        print("Done")
        return True
    else:
        return False

model_entorno = tf.keras.saving.load_model("../data/models/modelo_entorno.keras")
model_reward = tf.keras.saving.load_model("../data/models/modelo_reward.keras")

done = False

with open(f"../data/csv/history2.csv", 'w') as f:
    f.write("step,y,x,action,next_y,next_x,reward,done\n")
    for i in range(1):
        obs = np.array([[4, 0]]) #Starting position

        for t in range(50000):
            action = np.int32(np.random.randint(0, 4))

            prev_state = np.column_stack(np.array([obs[0][0], obs[0][1], action]))

            obs = model_entorno.predict(prev_state, verbose=0)
            rew = model_reward.predict(prev_state, verbose=0)
            #round the values
            obs = np.round(obs)
            rew = np.round(rew)

            done = check_done(obs)

            f.write(f"{t},{int(prev_state[0][0])},{int(prev_state[0][1])},{int(prev_state[0][2])},{int(obs[0][0])},{int(obs[0][1])},{int(rew[0])},{done}\n")

            if done:
                break

        print("Agente", i+1, "terminado en", t, "pasos")
