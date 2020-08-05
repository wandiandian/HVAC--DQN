from agent import DQNAgent
from utils import learning_curve
from HVAC import Building
from datetime import timedelta
import matplotlib.pyplot as plt

env = Building(heat_mass_capacity=16500000, heat_transmission=200,
               initial_building_temperature=28, initial_out_temperature=30.0,
               comfortable_temp_upper=26, comfortable_temp_lower=22, time_step_size=timedelta(minutes=1), )
agent = DQNAgent(env=env, capacity=10000, epochs=10, batch_size=64)  # env对应初始化中的env参数
max_episode_num = 1000
data = agent.learning(gamma=0.90,
                      epsilon=1,
                      decaying_epsilon=True,
                      alpha=0.1,
                      max_episode_num=max_episode_num,
                      display=False)

learning_curve(data, 2, 1, title="DQNAgent performance on PuckWorld",
               x_name="episodes", y_name="rewards of episode")
plt.plot(env.temperature[0:96])
plt.show()
plt.plot(env.temperature[96*max_episode_num - 96:96*max_episode_num])
plt.show()
plt.plot(env.temperature)
plt.show()
# plt.plot(env.temperature[286560:288000])
# plt.show()
# plt.plot(env.temperature[286560:288000:60])
# plt.show()
