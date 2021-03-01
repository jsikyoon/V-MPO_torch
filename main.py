import os
import torch
import numpy as np
from module import Memory, VMPO, PPO
from utils import set_up_hyperparams
from tensorboardX import SummaryWriter

import deepmind_lab

def main():

    H, logprint = set_up_hyperparams()
    tb_logger = SummaryWriter(H.save_dir)

    env = deepmind_lab.Lab(
            level='contributed/dmlab30/'+H.env_name,
            observations=['RGB_INTERLEAVED'],
            config={
                'width': '64',
                'height': '64',
                'logLevel': 'WARN',
                }
            )
    H.img_size = 64
    H.device = 'cuda:'+H.gpu if H.gpu is not None else 'cpu'

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
        env.reset()
        for t in range(H.max_timesteps):
            img = env.observations()['RGB_INTERLEAVED']
            timestep += 1

            # Running policy_old:
            action = agent.policy_old.act(t, img, memory)
            reward = env.step(H.action_list[action].astype(np.intc))

            if env.is_running():
                done = False
            else:
                done = True

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            running_reward += reward

            if not env.is_running():
                break

        # Update if its time
        if timestep > H.update_timestep:
            agent.update(memory)
            memory.clear_memory()
            timestep = 0

        avg_length += t

        # Logging
        if i_episode % H.log_interval == 0:
            avg_length = int(avg_length/H.log_interval)
            running_reward = int((running_reward/H.log_interval))

            logprint(model=H.desc,  type='tr_loss', episodes=i_episode,
                    **{'avg_length': avg_length, 'running_reward': running_reward})
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()
