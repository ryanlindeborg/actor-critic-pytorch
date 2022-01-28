import gym
import numpy as np
from actor_critic_agent import ActorCriticAgent

ENV_LUNAR_LANDER = "LunarLander-v2"

def run_ac_on_lunar_lander():
    #########################
    #### Hyperparameters ####
    #########################
    n_games = 2100

    env = gym.make(ENV_LUNAR_LANDER)
    actor_critic_agent = ActorCriticAgent(state_dims=[8], n_actions=4, lr=5e-6, gamma=0.99)

    episode_scores = []
    for episode_i in range(n_games):
        done = False
        state = env.reset()
        episode_score = 0
        while not done:
            action = actor_critic_agent.select_action(state)
            state_, reward, done, info = env.step(action)
            episode_score += reward
            actor_critic_agent.learn(state, reward, state_, done)
            state = state_
        episode_scores.append(episode_score)

        # Log trailing average last 100 episodes
        episode_score_trailing_average = np.mean(episode_scores[-100:])
        print(f"Episode: {episode_i}; Score: {episode_score}; Trailing average score: {episode_score_trailing_average}")




if __name__ == "__main__":
    run_ac_on_lunar_lander()