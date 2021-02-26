import torch
import gym
from module import Memory, VMPO, PPO
from utils import set_up_hyperparams

def main():

    H, logprint = set_up_hyperparams()

    env = gym.make(H.env_name)
    H.state_dim = env.observation_space.shape[0]
    H.device = 'cpu'

    memory = Memory()
    if H.model == 'vmpo':
        agent = VMPO(H)
    elif H.model == 'ppo':
        agent = PPO(H)

    # Logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # Training loop
    for i_episode in range(1, H.max_episodes+1):
        state = env.reset()
        for t in range(H.max_timesteps):
            timestep += 1

            # Running policy_old:
            action = agent.policy_old.act(t, state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            running_reward += reward
            if running_reward > (H.log_interval*H.solved_reward)*0.8:
                env.render()

            if done:
                break

        # Update if its time
        if timestep > H.update_timestep:
            agent.update(memory)
            memory.clear_memory()
            timestep = 0

        avg_length += t

        # Stop training if avg_reward > solved_reward
        if running_reward > (H.log_interval*H.solved_reward):
            print('######### Solved! ############')
            torch.save(agent.policy.state_dict(), './{}_{}_{}.pth'.format(model, state_representation, env_name))
            break

        # Logging
        if i_episode % H.log_interval == 0:
            avg_length = int(avg_length/H.log_interval)
            running_reward = int((running_reward/H.log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()
