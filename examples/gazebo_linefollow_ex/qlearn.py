import random
import pickle
import os

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        self.best_reward = float('-inf')

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        pickle_filename = filename + ".pickle"
        if os.path.exists(pickle_filename):
            try:
                with open(pickle_filename, 'rb') as file:
                    self.q = pickle.load(file)
                print(f"Loaded Q-values from {pickle_filename}")
                return True
            except Exception as e:
                print(e)
        return False

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        pickle_filename = filename + ".pickle"
        try:
            with open(pickle_filename, 'wb') as file:
                pickle.dump(self.q, file)
            print("Wrote to file: {}".format(pickle_filename))
        except Exception as e:
            print(f"Error saving Q-values: {e}")


    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_values = [self.getQ(state, act) for act in self.actions]
        max_q = max(q_values)
        
        best_actions = [act for act, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        current_q = self.getQ(state1, action1)
        max_next_q = max([self.getQ(state2, act) for act in self.actions]) if self.actions else 0
        
        updated_q_value = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q[(state1, action1)] = updated_q_value
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.saveQ("best_policy")

    def load_best_policy(self, filename="best_policy"):
        """
        @brief load the best known policy.
        """
        return self.loadQ(filename)