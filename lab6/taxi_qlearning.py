import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def initialize_q_table(state_space, action_space):
    """Initializes the Q-table with zeros."""
    return np.zeros((state_space, action_space))

def choose_action(env, q_table, state, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample() # Explore
    else:
        return np.argmax(q_table[state, :]) # Exploit

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """Updates the Q-table using the Bellman equation."""
    best_next_action = np.argmax(q_table[next_state, :])
    td_target = reward + gamma * q_table[next_state, best_next_action]
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    return q_table

def train_q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon):
    """Main training loop for the Q-Learning algorithm."""
    q_table = initialize_q_table(env.observation_space.n, env.action_space.n)
    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = choose_action(env, q_table, state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            
            q_table = update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
            
            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode

def run_experiments():
    """Runs experiments with different hyperparameters and plots the results."""
    env = gym.make("Taxi-v4")
    episodes = 2000
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    # Experimenting with different learning rates (alpha)
    learning_rates = [0.1, 0.5, 0.9]
    results = {}

    plt.figure(figsize=(10, 6))

    for alpha in learning_rates:
        print(f"Training with learning rate (alpha) = {alpha}...")
        _, rewards = train_q_learning(
            env, episodes, alpha, gamma, epsilon_start, epsilon_decay, min_epsilon
        )
        
        # Calculate moving average for smoother plots
        window = 100
        moving_avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        plt.plot(moving_avg_rewards, label=f'Alpha = {alpha}')

    plt.title('Q-Learning on Taxi-v3: Reward per Episode (100-Episode Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('taxi_training_results.png')
    print("Training complete. Results saved to 'taxi_training_results.png'.")
    plt.show()

if __name__ == "__main__":
    run_experiments()