# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import smooth
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def print_greedy_actions(Q):
    greedy_actions = np.argmax(Q, 1).reshape((12,12))
    print_string = np.zeros((12, 12), dtype=str)
    print_string[greedy_actions==0] = '^'
    print_string[greedy_actions==1] = 'v'
    print_string[greedy_actions==2] = '<'
    print_string[greedy_actions==3] = '>'
    print_string[np.max(Q, 1).reshape((12, 12))==0] = '0'
    line_breaks = np.zeros((12,1), dtype=str)
    line_breaks[:] = '\n'
    print_string = np.hstack((print_string, line_breaks))
    print(print_string.tobytes().decode('utf-8'))
    
# We applied dynamically smoothing approach to adjust the window size based on the current position in the data series
def dynamic_smooth(y, start_window=5, poly=3):
    start_window = min(start_window if start_window % 2 != 0 else start_window - 1, len(y) - 1)
    smoothed = np.zeros_like(y)
    
    for i in range(len(y)):
        if i < start_window // 2:
            current_window = 2 * i + 1
        elif len(y) - i <= start_window // 2:
            current_window = 2 * (len(y) - i - 1) + 1
        else:
            current_window = start_window
        if current_window >= poly + 2:  
            smoothed_segment = savgol_filter(y[max(0, i - current_window // 2):i + current_window // 2 + 1], current_window, poly)
            smoothed[i] = smoothed_segment[min(i, current_window // 2)]
        else:
            smoothed[i] = y[i] 
    
    return smoothed    
    
################################################################   Q-LEARNING    ###############################################################   

def run_experiments(alpha, epsilon=0.1, n_episodes=1000, n_rep=100):

    env = ShortcutEnvironment()
    cumulative_rewards = np.zeros((n_rep, n_episodes))

    for rep in range(n_rep):
        agent = QLearningAgent(env.action_size(), env.state_size(), epsilon, alpha)
        total_reward = 0

        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
            
            total_reward += episode_reward
            cumulative_rewards[rep, episode] = total_reward / (episode + 1)

    # Averaging over repetitions
    avg_cumulative_rewards = np.mean(cumulative_rewards, axis=0)
    return avg_cumulative_rewards

def run_repetitions(alpha=0.1, epsilon=0.1, n_episodes=1000, n_rep=100):
    cumulative_rewards = np.zeros(n_episodes)
    
    for rep in range(n_rep):
        env = ShortcutEnvironment()
        agent = QLearningAgent(env.action_size(), env.state_size(), epsilon, alpha)
        total_reward = 0
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            done = env.done()
            
            while not done:
                action = agent.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                agent.update(state, action, reward, next_state)
                episode_reward += reward
                state = next_state
            
            total_reward += episode_reward
            cumulative_rewards[episode] += episode_reward
    
    average_cumulative_rewards = cumulative_rewards / n_rep
    return average_cumulative_rewards

def plot_different_alphas(alphas, epsilon=0.1, n_episodes=1000, n_rep=100):
    plt.figure(figsize=(10, 6))
    
    for alpha in alphas:
        avg_rewards = run_repetitions(alpha, epsilon, n_episodes, n_rep)
        smoothed_rewards = dynamic_smooth(avg_rewards, start_window=9, poly=3)
        plt.plot(smoothed_rewards, label=f'α = {alpha}')
        
        # Print greedy actions only for alpha = 0.1
        if alpha == 0.1:
            env = ShortcutEnvironment()  # Reset environment for a clean start
            agent = QLearningAgent(env.action_size(), env.state_size(), epsilon, alpha)
            for episode in range(n_episodes):
                state = env.reset()
                done = env.done()

                while not done:
                    action = agent.select_action(state)
                    reward = env.step(action)
                    next_state = env.state()
                    done = env.done()
                    agent.update(state, action, reward, next_state)
                    state = next_state

            # After the final episode, we print the greedy actions
            print(f"Greedy path for alpha = {alpha}:")
            print_greedy_actions(agent.Q)
    
    plt.title('Q-Learning Learning Curves for Different Alphas')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig('Q_learning_curve.png')
    plt.show()

# Alphas to test
alphas = [0.01, 0.1, 0.5, 0.9]
plot_different_alphas(alphas)


######################################################################      SARSA      #######################################################################

def run_single_experiment_sarsa(alpha=0.1, epsilon=0.1, n_episodes=10000):
    env = ShortcutEnvironment()
    agent = SARSAAgent(env.action_size(), env.state_size(), epsilon, alpha)
    for episode in range(n_episodes):
        state = env.reset()
        action = agent.select_action(state)
        done = env.done()
        while not done:
            reward = env.step(action)
            next_state = env.state()
            next_action = agent.select_action(next_state) if not env.done() else None
            agent.update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            done = env.done()
    print_greedy_actions(agent.Q)

def run_repetitions_sarsa(alpha=0.1, epsilon=0.1, n_episodes=1000, n_rep=100):
    env = ShortcutEnvironment()
    cumulative_rewards = np.zeros((n_rep, n_episodes))

    for rep in range(n_rep):
        agent = SARSAAgent(env.action_size(), env.state_size(), epsilon, alpha)
        total_reward = 0

        for episode in range(n_episodes):
            state = env.reset()
            action = agent.select_action(state)
            done = env.done()
            episode_reward = 0

            while not done:
                reward = env.step(action)
                next_state = env.state()
                if not env.done():
                    next_action = agent.select_action(next_state)
                else:
                    next_action = None
                agent.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                episode_reward += reward
                done = env.done()

            total_reward += episode_reward
            cumulative_rewards[rep, episode] = total_reward / (episode + 1)

    # Averaging over repetitions
    avg_cumulative_rewards = np.mean(cumulative_rewards, axis=0)
    return avg_cumulative_rewards

def plot_different_alphas_sarsa(alphas, epsilon=0.1, n_episodes=1000, n_rep=100):
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        avg_rewards = run_repetitions_sarsa(alpha, epsilon, n_episodes, n_rep)
        smoothed_rewards = dynamic_smooth(avg_rewards, start_window=9, poly=3)
        plt.plot(smoothed_rewards, label=f'α = {alpha}')
        
    plt.title('SARSA Learning Curves for Different Alphas')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig('SARSA_learning_curves.png')
    plt.show()


# Running the SARSA experiments with specified alphas
alphas = [0.01, 0.1, 0.5, 0.9]
plot_different_alphas_sarsa(alphas)

########################################################################## STORMY WEATHER ############################################################################

def run_single_experiment_qlearning(alpha=0.1, epsilon=0.1, n_episodes=10000):
    env = WindyShortcutEnvironment()
    agent = QLearningAgent(env.action_size(), env.state_size(), epsilon, alpha)
    cumulative_rewards = np.zeros(n_episodes)  # Track cumulative rewards for plotting

    for episode in range(n_episodes):
        state = env.reset()
        done = env.done()
        total_reward = 0  # Track total reward for this episode

        while not done:
            action = agent.select_action(state)
            reward = env.step(action)
            next_state = env.state()
            done = env.done()
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward  # Accumulate rewards
        
        cumulative_rewards[episode] = total_reward  # Store the total reward

    print_greedy_actions(agent.Q)

def run_single_experiment_sarsa(alpha=0.1, epsilon=0.1, n_episodes=10000):
    env = WindyShortcutEnvironment()
    agent = SARSAAgent(env.action_size(), env.state_size(), epsilon, alpha)
    cumulative_rewards = np.zeros(n_episodes)  # Track cumulative rewards for plotting

    for episode in range(n_episodes):
        state = env.reset()
        action = agent.select_action(state)
        done = env.done()
        total_reward = 0  # Track total reward for this episode

        while not done:
            reward = env.step(action)
            next_state = env.state()
            next_action = agent.select_action(next_state) if not env.done() else None
            agent.update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            done = env.done()
            total_reward += reward  
        
        cumulative_rewards[episode] = total_reward  # Store the total reward

    print_greedy_actions(agent.Q)
    
run_single_experiment_qlearning(alpha=0.1, epsilon=0.1, n_episodes=10000)
run_single_experiment_sarsa(alpha=0.1, epsilon=0.1, n_episodes=10000)

######################################################################## EXPECTED SARSA ############################################################################

def run_single_experiment_expected_sarsa(alpha=0.1, epsilon=0.1, n_episodes=10000):
    env = ShortcutEnvironment()
    agent = ExpectedSARSAAgent(env.action_size(), env.state_size(), epsilon, alpha)
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0  
        while not env.done():
            action = agent.select_action(state)
            reward = env.step(action)  
            next_state = env.state()
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
    
def run_repetitions_expected_sarsa(alpha, epsilon=0.1, n_episodes=1000, n_rep=100):
    env = ShortcutEnvironment()
    cumulative_rewards = np.zeros((n_rep, n_episodes))
    for rep in range(n_rep):
        agent = ExpectedSARSAAgent(env.action_size(), env.state_size(), epsilon, alpha)
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            while not env.done():
                action = agent.select_action(state)
                reward = env.step(action)  
                next_state = env.state()
                agent.update(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
            cumulative_rewards[rep, episode] = episode_reward
    avg_cumulative_rewards = np.mean(cumulative_rewards, axis=0)
    return avg_cumulative_rewards

def plot_different_alphas_expected_sarsa(alphas, epsilon=0.1, n_episodes=1000, n_rep=100):
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        avg_rewards = run_repetitions_expected_sarsa(alpha, epsilon, n_episodes, n_rep)
        
        smoothed_rewards = dynamic_smooth(avg_rewards, start_window=9, poly=3)
        plt.plot(smoothed_rewards, label=f'α = {alpha}')
    plt.title('Expected SARSA Learning Curves for Different Alphas')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig('Expected_SARSA_learning_curves.png')
    plt.show()

alphas = [0.01, 0.1, 0.5, 0.9]
plot_different_alphas_expected_sarsa(alphas)


