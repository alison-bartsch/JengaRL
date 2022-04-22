import gym
from jenga_discrete import JengaEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env


# Parallel environments
env = JengaEnv()

model = A2C("MlpPolicy", env, verbose=1)

print("Beginning training...")
model.learn(total_timesteps=250)
print("Ended training...")
model.save("a2c_jenga")

# del model # remove to demonstrate saving and loading
# model = A2C.load("a2c_jenga")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()