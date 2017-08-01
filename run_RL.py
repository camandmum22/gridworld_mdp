# coding=utf-8
import numpy as np
import board
import random


class Value_Iteration(object):
    def __init__(self, states, actions, transitions, rewards, discount, epsilon):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount = discount
        self.epsilon = epsilon
        self.utilities_base = np.zeros(self.states)
        self.utilities = []
        self.optimal_policy = -np.ones(self.states)
        self.run()

    def run(self):
        x = 0
        while True:
            x += 1
            self.utilities = np.copy(self.utilities_base)
            diff = 0
            for state in range(self.states):
                max_utility = - float('inf')
                opt_action = -1
                for action in range(self.actions):
                    current_utility = np.sum(np.multiply(self.transitions[state, :, action], self.utilities))

                    if current_utility > max_utility:
                        max_utility = current_utility
                        opt_action = action

                # Update utilities_base and optimal_policy
                self.utilities_base[state] = self.rewards[state] + self.discount * max_utility
                self.optimal_policy[state] = opt_action

                if abs(self.utilities_base[state] - self.utilities[state]) > diff:
                    diff = self.utilities_base[state] - self.utilities[state]

            if diff <= self.epsilon:
                break

class Q_Learning(object):
    def __init__(self, states, actions, transitions, rewards, init_state, episodes, discount, epsilon):
        self.states = states
        self.actions = actions
        self.Q_values = np.zeros((self.states, self.actions))
        # maps actions and states to indicate how amny times the action have been visited in that state
        self.visits = np.zeros((self.states, self.actions))
        self.transitions = transitions
        self.rewards = rewards
        self.init_state = init_state
        self.limit_episodes = episodes
        self.discount = discount
        self.epsilon = epsilon
        self.utilities = -np.ones(self.states)
        self.optimal_policy = -np.ones(self.states)
        self.run()

    def run(self):
        for i in range(self.limit_episodes):
            state = self.init_state
            while state != 16: # terminal state
                a = self.choose_action(state)
                self.visits[state, a] += 1
                set_prob = self.transitions[state, :, a]
                sorted_prob = np.sort(set_prob[set_prob > 0])
                # imitate the transition model
                num_random = random.random()
                # middle of the board --> 3 potential moves/actions
                if len(sorted_prob) >3 :
                    print(len(sorted_prob))
                if len(sorted_prob) == 3:
                    if (num_random < 0.9):
                        s_2 = np.where(set_prob == 0.9)[0][0]
                    elif (num_random < 0.95):
                        s_2 = np.where(set_prob == 0.05)[0][0]
                    else:
                        s_2 = np.where(set_prob == 0.05)[0][1]
                # edge of the board --> 2 potential moves/actions
                elif len(sorted_prob) == 2:
                    if (num_random < 0.95):
                        s_2 = np.where(set_prob == max(set_prob))[0][0]
                    else:
                        s_2 = np.where(set_prob == 0.05)[0][0]
                # pre-last state (15) --> only one potential move/action
                elif len(sorted_prob) == 1:
                    s_2 = np.where(set_prob == 1)[0][0]

                visited = 1 / self.visits[state, a]
                self.Q_values[state, a] = self.Q_values[state, a] + visited * (self.rewards[state] + self.discount * max(self.Q_values[s_2, :]) - self.Q_values[state, a])
                state = s_2

        # Update utilities and optimal_policy
        for state in range(self.states):
            self.utilities[state] = max(self.Q_values[state])
            self.optimal_policy[state] = np.argmax(self.Q_values[state])

    def choose_action(self, state):
        num_random = random.random()
        actions = list(range(self.actions))
        if num_random < self.epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(self.Q_values[state])
        return action


def main():
    # np.random.seed(0)
    states = 17
    actions = 4
    np.set_printoptions(suppress=True)
    a_cases = [0.9, 0.8]
    b_cases = [0.05, 0.1]
    discount = 0.99
    epsilon = 0.01
    epsilon_QL = [0.05, 0.2]
    max_iter_QL = 10000
    np.set_printoptions(threshold=np.inf)

    TRANSITION_1, REWARD_1 = board.gridWorld(a=a_cases[0], b=b_cases[0])
    TRANSITION_2, REWARD_2 = board.gridWorld(a=a_cases[1], b=b_cases[1])

    vat_1 = Value_Iteration(states=states, actions=actions,transitions=TRANSITION_1, rewards=REWARD_1, discount=discount,epsilon=epsilon)
    vat_2 = Value_Iteration(states=states, actions=actions,transitions=TRANSITION_2, rewards=REWARD_2, discount=discount, epsilon=epsilon)
    qle_1 = Q_Learning(states=states, actions=actions,transitions=TRANSITION_1, rewards=REWARD_1, init_state = actions, episodes=max_iter_QL, discount=discount,epsilon=epsilon_QL[0])
    qle_2 = Q_Learning(states=states, actions=actions,transitions=TRANSITION_1, rewards=REWARD_1, init_state=actions, episodes=max_iter_QL, discount=discount,epsilon=epsilon_QL[1])

    print('Value Iteration with a = %s and b = %s\n%s\n%s' % (a_cases[0],b_cases[0], vat_1.utilities[:16], vat_1.optimal_policy[:16]))
    print('Value Iteration with a = %s and b = %s\n%s\n%s' % (a_cases[1],b_cases[1], vat_2.utilities[:16], vat_2.optimal_policy[:16]))
    print('Q Learning with epsilon = %s\n%s\n%s' % (epsilon_QL[0], qle_1.utilities[:16], qle_1.optimal_policy[:16]))
    print('Q Learning with epsilon = %s\n%s\n%s' % (epsilon_QL[1], qle_2.utilities[:16], qle_2.optimal_policy[:16]))

if __name__ == "__main__":
    main()
