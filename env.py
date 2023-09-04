import random
import config
from agent import Seller_Agent, Buyer_Agent
from copy import deepcopy
import json
import time
from plot import plot_rewards, plot_actions, plot_dif

cold_period = 1000
iterations = 2000

agent = Seller_Agent(p_he=0.8, p_le=0.2, cost_0=0.1, max_time=1, learning_rate=0.1, exploration_rate=0.3,
                     iterations=iterations - cold_period, period=30)
agent.reset()
agent.init_q_table()

total_reward = 0  # Score keeping
reward_list_seller = []
reward_ave_list_seller = []
dif_list_seller = []
dif_ave_list_seller = []
last_total = 0
action_list_seller = []
reward_ave = 0
dif_ave = 0

## set buyer
buyer = Buyer_Agent(learning_rate=0.1, exploration_rate=0.3, iterations=iterations - cold_period)
buyer.reset()
buyer.init_q_table()
total_reward_buyer = 0  # Score keeping
reward_list_buyer = []
reward_ave_list_buyer = []
dif_list_buyer = []
dif_ave_list_buyer = []
last_total_buyer = 0
action_list_buyer = []
reward_ave_buyer = 0
dif_ave_buyer = 0

# main loop
for step in range(cold_period):  # 冷启动
    old_state = agent.state  # Store current state
    action = agent.random_action()  # Query agent for the next action
    old_state_buyer = buyer.state  # Store current state
    action_buyer = buyer.random_action()  # Query agent for the next action
    new_state, quality, reputation, cost = agent.take_action(action, step)
    new_state_buyer, price = buyer.take_action(action, signal=reputation, t=step)
    reward = agent.get_reward(price=price)
    reward_buyer = buyer.get_reward(price, signal=reputation)
    dif = agent.update(old_state, new_state, action, reward)  # Let the agent update internals
    dif_buyer = buyer.update(old_state_buyer, new_state_buyer, action_buyer, reward_buyer)

for step in range(cold_period, iterations):
    old_state = agent.state  # Store current state
    action = agent.get_next_action(old_state)  # Query agent for the next action
    old_state_buyer = buyer.state  # Store current state
    action_buyer = buyer.get_next_action(old_state_buyer)  # Query agent for the next action
    new_state, quality, reputation, cost = agent.take_action(action, step)
    new_state_buyer, price = buyer.take_action(action, signal=reputation, t=step)
    reward = agent.get_reward(price)
    reward_buyer = buyer.get_reward(price, signal=reputation)
    dif = agent.update(old_state, new_state, action, reward)  # Let the agent update internals
    dif_buyer = buyer.update(old_state_buyer, new_state_buyer, action_buyer, reward_buyer)
    total_reward += reward  # Keep score
    reward_list_seller.append(reward)
    action_list_seller.append(action)
    dif_list_seller.append(dif)

    total_reward_buyer += reward_buyer  # Keep score
    reward_list_buyer.append(reward_buyer)
    action_list_buyer.append(action_buyer)
    dif_list_buyer.append(dif_buyer)

    interval = 100
    if step % interval != 0:
        reward_ave += reward
        dif_ave += dif

        reward_ave_buyer += reward_buyer
        dif_ave_buyer += dif_buyer
    else:
        reward_ave_list_seller.append(reward_ave / interval)
        reward_ave = 0
        dif_ave_list_seller.append(dif_ave / interval)
        dif_ave = 0

        reward_ave_list_buyer.append(reward_ave_buyer / interval)
        reward_ave_buyer = 0
        dif_ave_list_buyer.append(dif_ave_buyer / interval)
        dif_ave_buyer = 0

    interval_1 = 10
    if step % interval_1 == 0:  # Print out metadata every 250th iteration
        performance = (total_reward - last_total) / interval_1
        print('step:{},performance_seller:{},total_reward_seller:{},action_seller:{}'.format(step, performance, total_reward, action))
        last_total = total_reward
        # time.sleep(0.0001)  # Avoid spamming stdout too fast!
        q_table_decode = agent.q_table.reshape((agent.state_space, agent.action_space))
        # print(q_table_decode)

        performance_buyer = (total_reward_buyer - last_total_buyer) / interval_1
        print('step:{},performance_buyer:{},total_reward_buyer:{},action_buyer:{}'.format(step, performance_buyer, total_reward_buyer, action_buyer))
        last_total_buyer = total_reward_buyer
        # time.sleep(0.0001)  # Avoid spamming stdout too fast!
        q_table_decode_buyer = buyer.q_table.reshape((buyer.state_space, buyer.action_space))
        # print(q_table_decode_buyer)
    plot_dif(dif_ave_list_seller, postfix="seller")
    plot_rewards(reward_ave_list_seller, postfix="seller")
    plot_actions(action_list_seller, postfix="seller")
    plot_dif(dif_ave_list_buyer, postfix="buyer")
    plot_rewards(reward_ave_list_buyer, postfix="buyer")
    plot_actions(action_list_buyer, postfix="buyer")
