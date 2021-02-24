import torch
import gym
from module import Memory, VMPO, PPO

def main():
    ################# Hyperparameters ##########################
    model = 'ppo' # {vmpo|ppo}
    state_representation = 'lstm' # {none|lstm|trxl|gtrxl}
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 230 # stop training if avg_reward > solved_reward
    log_interval = 20 # print avg reward in the interval
    max_episodes = 50000 # max training episodes
    max_timesteps = 300 # max timesteps in one episode
    n_latent_var = 64 # number of variables in hidden layer
    update_timestep = 2400 # update policy every n timesteps
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99 # discount factor
    K_epochs = 8 # update policy for K epochs
    eps_clip = 0.2 # clip parameter for PPO
    random_seed = None
    gpu = None
    device = "cuda:"+str(gpu) if gpu is not None else "cpu"
    ############################################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    if model == 'vmpo':
        agent = VMPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, state_representation, device)
    elif model == 'ppo':
        agent = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, state_representation, device)

    # Logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # Training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = agent.policy_old.act(t, state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            running_reward += reward
            if render or running_reward > (log_interval*solved_reward)*0.8:
                env.render()

            if done:
                break

        # Update if its time
        if timestep > update_timestep:
            agent.update(memory)
            memory.clear_memory()
            timestep = 0

        avg_length += t

        # Stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print('######### Solved! ############')
            torch.save(agent.policy.state_dict(), './{}_{}_{}.pth'.format(model, state_representation, env_name))
            break

        # Logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()
