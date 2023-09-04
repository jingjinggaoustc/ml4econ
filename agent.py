import numpy as np
from copy import deepcopy
import random
from sklearn import linear_model


# seller agent


class Seller_Agent:

    def __init__(self, p_he, p_le, cost_0, max_time, learning_rate, exploration_rate,
                 iterations, period=30):
        self.cost = []  # cost of invest
        self.cost_0 = cost_0
        self.quality = 0  # the quality of the product
        self.num = []
        self.p_he = p_he
        self.p_le = p_le
        self.max_time = max_time
        self.reputation = []  # the reputation of seller, sigma t
        self.state = 0
        self.action_history = []  # the history of action of seller -> invest or not
        self.reward_history = []  # the reward history ： reward = price(sigma_t)-cost(a_t;sigma_t})
        self.quality_history = []
        self.period = period
        self.state_space = period + 1
        self.action_space = 2  # invest or not invest
        self.q_table = []
        self.learning_rate = learning_rate  # How much we appreciate new q-value over current
        # self.discount = discount  # How much we appreciate future reward over current
        self.discount = pow(1e-10, 1 / self.period)
        self.exploration_rate = exploration_rate  # Initial exploration rate
        self.exploration_delta = exploration_rate / iterations  # Shift from exploration to explotation

    ##
    def init_q_table(self):
        self.q_table = np.zeros(self.state_space * self.action_space)

    def reset(self):
        self.state = 0  # Reset state to zero, the beginning of dungeon
        return self.state

    def get_next_action(self, state):
        if random.random() > self.exploration_rate:  # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()

    def random_action(self):
        action = random.randint(0, self.action_space - 1)
        return action

    def greedy_action(self, state):
        state_action = []
        for m in range(self.action_space):
            state_action.append(self.q_table[state * self.action_space + m])
        array = np.array(state_action)
        action = array.argmax()
        return action

    def update(self, old_state, new_state, action, reward):
        # Old Q-table value
        # old_value = self.q_table[action][old_state]
        old_value = self.q_table[old_state * self.action_space + action]
        # What would be our best next action?
        future_action = self.greedy_action(new_state)
        # What is reward for the best next action?
        # future_reward = self.q_table[future_action][new_state]
        future_reward = self.q_table[new_state * self.action_space + future_action]
        # Main Q-table updating algorithm
        new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
        # self.q_table[action][old_state] = new_value
        self.q_table[old_state * self.action_space + action] = new_value
        dif = new_value - old_value
        # Finally shift our exploration_rate toward zero (less gambling)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta
        return dif

    def cal_reputation(self, rand, p, t):
        if rand <= p:
            quality = 1  # high quality
        else:
            quality = 0  # low quality
        self.quality_history.append(quality)
        if (t - self.period) >= 0:
            quality_list = self.quality_history[t - self.period:t]
        else:
            quality_list = self.quality_history[0:t]
        if len(quality_list) == 0:
            x = 0 if random.random() < 0.5 else 1
            quality_list.append(x)
        n_h = quality_list.count(1)
        n_l = quality_list.count(0)
        reputation = n_h / (n_h + n_l)
        return quality, reputation

    def get_reward(self, price):
        # 出价函数
        reward = price - self.cost
        return reward

    def take_action(self, action, t):
        if action == 1:  # seller choose to invest
            self.cost = self.cost_0
            quality, reputation = self.cal_reputation(random.random(), self.p_he, t)
        else:  # seller choose not to invest
            self.cost = 0
            quality, reputation = self.cal_reputation(random.random(), self.p_le, t)
        self.state = int(reputation * self.period)
        self.quality = quality
        # price = self.bid_price(reputation)
        # reward = price-cost
        return self.state, quality, reputation, self.cost


class Buyer_Agent:

    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.state = 0  # the expected reputation
        self.action = 0  # bid of price
        self.expected_reputation = 0
        self.signal_history = []  # reputation history of seller
        self.action_history = []  # the history of action of buyer
        self.reward_history = []  # the reward history
        self.action_space = 3  # the range of price 0,1,2,3
        self.state_space = 100  # the range of sigma
        self.q_table = []
        self.learning_rate = learning_rate  # How much we appreciate new q-value over current
        self.discount = discount  # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate  # Initial exploration rate
        self.exploration_delta = exploration_rate / iterations  # Shift from exploration to explotation

    ##
    def init_q_table(self):
        self.q_table = np.zeros(self.state_space * self.action_space)

    def get_signal(self, signal):
        self.signal_history.append(signal)

    def get_expected_reputation(self, t):
        if t < 2:
            expected_reputation = self.signal_history[t]
        else:
            lm = linear_model.LinearRegression()
            x = self.signal_history[0:t]
            x = np.array(x)
            x = x.reshape(-1, 1)
            y = deepcopy(self.signal_history[1:t + 1])
            y = np.array(y)
            y = y.reshape(-1, 1)
            model = lm.fit(x, y)
            x_temp = self.signal_history[-1::]
            x_temp = np.array(x_temp)
            x_temp = x_temp.reshape(1, -1)
            y_temp = model.predict(x_temp)
            expected_reputation = y_temp[0][0]
        return expected_reputation

    def get_state(self, t):
        self.expected_reputation = self.get_expected_reputation(t)
        state = int(self.expected_reputation * (self.state_space-1))
        return state

    def reset(self):
        self.state = 0  # Reset state to zero, the beginning of dungeon
        return self.state

    def get_next_action(self, state):
        if random.random() > self.exploration_rate:  # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()

    def random_action(self):
        action = random.randint(0, self.action_space - 1)
        return action

    def greedy_action(self, state):
        state_action = []
        for m in range(self.action_space):
            state_action.append(self.q_table[state * self.action_space + m])
        array = np.array(state_action)
        action = array.argmax()
        return action

    def update(self, old_state, new_state, action, reward):
        old_value = self.q_table[old_state * self.action_space + action]
        future_action = self.greedy_action(new_state)
        future_reward = self.q_table[new_state * self.action_space + future_action]
        new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
        self.q_table[old_state * self.action_space + action] = new_value
        dif = new_value - old_value
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta
        return dif

    def get_reward(self, price, signal):
        reward = 2 * signal - price - 0.5*(abs(signal - self.expected_reputation))
        return reward

    def take_action(self, action, signal, t):
        self.get_signal(signal)
        self.state = self.get_state(t)
        price = action / self.action_space  # 归一化在0-1之间
        return self.state, price
