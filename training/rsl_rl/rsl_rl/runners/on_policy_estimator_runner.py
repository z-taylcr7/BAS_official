# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn as nn
import numpy as np

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticEstimator
from rsl_rl.env import VecEnv


class OnPolicyEstimatorRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.is_learning_policy = True
        # self.proception_dims = torch.cat([torch.arange(0, 13), torch.arange(14, 50)])
        self.proception_dims = torch.arange(0, 50)
        self.proception_dim = len(self.proception_dims)
        num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        self.policy_cfg["latent_dim"],
                                                        self.proception_dim,
                                                        self.policy_cfg["estimated_dim"],
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.use_estimation = self.policy_cfg["use_estimation"]
        self.history_length = self.policy_cfg["history_length"]
        self.estimated_dim = self.policy_cfg["estimated_dim"]
        self.latent_dim = self.policy_cfg["latent_dim"]
        self.critic_dim = self.env.num_obs+self.estimated_dim
        self.actor_dim = self.env.num_obs+self.estimated_dim

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.actor_dim], [self.critic_dim], [self.env.num_actions])

        if self.policy_cfg["use_estimation"]:
            self.estimate_estimated_dim = self.estimated_dim
            # self.privileged_obs_estimator = nn.Sequential(nn.Linear(self.policy_cfg['history_length']*len(self.proception_dims), 64),nn.ReLU(), nn.Linear(64,64),nn.ReLU(),nn.Linear(64,self.estimate_estimated_dim)).to(self.device)
            self.privileged_obs_estimator = Estimator(input_dim=len(self.proception_dims),
            history_length=self.history_length, 
            output_dim=self.estimated_dim, 
            nn_type='mlp').to(self.device)
            
            if self.policy_cfg["implicit_estimation"]: # if True, PPO will update the estimator
                if self.policy_cfg["learn_estimation"]:
                    self.alg.optimizer = torch.optim.Adam(list(self.alg.actor_critic.parameters())+list(self.privileged_obs_estimator.parameters()), lr=2e-5)  
                else:
                    self.privileged_obs_estimator.model = torch.load("legged_gym/logs/go1_pos_rough/your_ckpt.pt")
                    self.privileged_obs_estimator.model.eval()
            else: # Stop gradient between PPO and estimator
                if self.policy_cfg["learn_estimation"]:
                    self.privileged_obs_estimator.train()
                    self.privileged_obs_optimizer = torch.optim.Adam(self.privileged_obs_estimator.parameters(), lr=2e-5)
                    self.privileged_obs_optimizer.zero_grad() 
                else:# override a fixed estimator
                    self.privileged_obs_estimator.model = torch.load("legged_gym/logs/go1_pos_rough/your_ckpt.pt")
                    self.privileged_obs_estimator.model.eval()
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        if self.policy_cfg["use_estimation"]: # using estimator means you need to record history
            obs_history = torch.zeros((self.env.num_envs, self.history_length, len(self.proception_dims)), device=self.device)
            print(f"History length: {self.history_length}")
        # critic_obs = privileged_obs if privileged_obs is not None else obs
        latent = torch.zeros((self.env.num_envs, self.latent_dim), device=self.device)
        critic_obs = obs 
        ep_infos = []
        l_recon = 0.5

        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        privileged_obs = privileged_obs.to(self.device) if privileged_obs is not None else None
        if self.policy_cfg["use_estimation"]:
            privileged_obs = self.privileged_obs_estimator(obs_history.clone()).squeeze(0)
            obs = torch.cat((obs, privileged_obs), dim=-1)
            critic_obs = torch.cat((critic_obs, privileged_obs), dim=-1)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        # self.alg.actor_critic.eval()
        if self.policy_cfg["use_estimation"]:
            self.privileged_obs_estimator.train()
        rewbuffer = deque(maxlen=100)
        costbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        last_estimation = torch.zeros(self.env.num_envs, self.estimated_dim, device=self.device)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_cost_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # pr_done = False # for early stopping

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            if it % 2 == 0: # 2 is a compromise to memory...
                obs_history_buffer = []
                privileged_obs_buffer = []
            if not self.policy_cfg["use_estimation"]:
                it_rate = 0.0
            else:
                it_rate = (it-self.current_learning_iteration)/(tot_iter-self.current_learning_iteration) # linear fusion, + fine-tuning in last 5000 iterations
                # it_rate = 1 # supervised 
                # it_rate = 0 # ground-truth
                it_rate = min(1.0, it_rate) # This ensures that `it_rate` does not exceed 1.0.
            print(f"iteration alpha rate: {it_rate}")
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    privileged_obs = privileged_obs.squeeze(0).to(self.device) if privileged_obs is not None else None
                    self.alg.process_env_step(rewards, dones, infos)
                    if it % 20 == 0:
                        obs[:, 10:13] = 0 # standing still data augmentation. Don't learn at these epochs.
                    if self.policy_cfg["use_estimation"]:
                        if self.policy_cfg["implicit_estimation"]:
                            estimated_privileged_obs = self.privileged_obs_estimator(obs_history.clone()).squeeze(0)
                            # print(f"Estimated privileged obs: {estimated_privileged_obs}")
                            obs = torch.cat((obs, estimated_privileged_obs), dim=-1)
                            critic_obs = torch.cat((critic_obs, estimated_privileged_obs), dim=-1)
                        else: 
                            estimated_privileged_obs = self.privileged_obs_estimator(obs_history.clone()).squeeze(0).detach().clone()
                            # last_est_tensor = torch.tensor(last_estimation, device=self.device)
                            # curr_est_tensor = torch.tensor(estimated_privileged_obs, device=self.device)
                            # estimation_rate = torch.sum(torch.sum(torch.square(last_est_tensor-curr_est_tensor),dim=0))
                            # last_estimation = estimated_privileged_obs.detach().clone()
                            # print("ER: ", estimation_rate)

                            # fusion interpolation
                            fusion_privileged_obs = (estimated_privileged_obs*(it_rate)+privileged_obs*(1-it_rate))
                            obs = torch.cat((obs, fusion_privileged_obs), dim=-1)
                            critic_obs = torch.cat((critic_obs, privileged_obs), dim=-1)
                            # calculate mean squared error
                            mse_loss = torch.nn.functional.mse_loss(estimated_privileged_obs, privileged_obs.clone())
                            if it % 1000 == 0:
                                print(f"MSE error of estimation: {mse_loss.item()}")
                    if self.policy_cfg["use_estimation"]: #update history
                        obs_history = torch.cat((obs_history[:, 1:, :], obs[:,self.proception_dims].unsqueeze(1)), dim=1)
                    
                    # print('privi_obs:', privileged_obs)
                    # import pdb; pdb.set_trace()
                    if self.policy_cfg["learn_estimation"]:#randomly put (history,env factor) into dataset
                        # where_available = torch.where(torch.mean(torch.abs(obs[:,10:12]))>=0.08)
                        rnd_idx = np.random.randint(self.env.num_envs/2, self.env.num_envs, size=128) # 128 is a compromise to GPU memory...
                        obs_history_buffer.append(obs_history[rnd_idx].clone())
                        privileged_obs_buffer.append(privileged_obs[rnd_idx].clone())
                        # obs_history_buffer.append(obs_history.clone())
                        # privileged_obs_buffer.append(privileged_obs.clone())
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_cost_sum += infos['cost']
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        costbuffer.extend(cur_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_cost_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.policy_cfg["use_estimation"]:
                            obs_history[new_ids,:,:] = 0
                    
                        

                
                stop = time.time()
                collection_time = stop - start
                # Learning step
                start = stop
                
                self.alg.compute_returns(critic_obs)
            
             
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            # mean_value_loss, mean_surrogate_loss = 0,0
            
            if self.policy_cfg["learn_estimation"] and len(obs_history_buffer) > 0:
                for privi_obs_tensor, obs_history_tensor in zip(privileged_obs_buffer, obs_history_buffer):
                    privi_obs_tensor.requires_grad = False
                    estimated_privileged_obs = self.privileged_obs_estimator(obs_history_tensor.clone())
                    self.privileged_obs_optimizer.zero_grad()
                    # privi_loss = (estimated_privileged_obs.flatten()-(privi_obs_tensor.flatten()/5.0-torch.ones_like(privi_obs_tensor.flatten())))**2 # normalization
                    privi_loss = (estimated_privileged_obs-(privi_obs_tensor))**2
                    privi_loss = torch.sum(privi_loss,dim=1)
                    mse_loss = torch.mean(privi_loss,dim=0)
                    
                    lambda_reg = 1e-5  # l2 regularization
                    l2_reg = torch.tensor(0.).to(estimated_privileged_obs.device)
                    for param in self.privileged_obs_estimator.model.parameters():
                        l2_reg += torch.norm(param)**2
                    privi_loss = torch.mean(privi_loss,dim=0) + 0.5 * lambda_reg * l2_reg
                    privi_loss.backward()
                    self.privileged_obs_optimizer.step()
                    if np.random.rand() < 0.001:
                        print(f"Privileged loss: {privi_loss.item()}")
                
            else:
                privi_loss = -1.0
                mse_loss = -1.0
            stop = time.time()

            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

            # Online changing mass, not applicable in isaac while training
            # if it % 2 == 0 and it > 0:
            #     # if self.env.env_cfg.domain_rand.randomize_base_mass:
            #     # self.envs_loads[env_ids] = self.envs_loads[env_ids].uniform_(self.cfg.domain_rand.min_base_mass, self.cfg.domain_rand.max_base_mass)
            #     for j in range(self.env.num_envs):
            #         env_handle = self.env.envs[j]
            #         actor_handle = self.env.actor_handles[j]
            #         body_props = self.env.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            #         body_props = self.env._process_rigid_body_props(body_props, j)
                    
            #         # self.env.envs[j] = env_handle
            #         # self.env.actor_handles[j] = actor_handle
            #         self.env.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
                                    
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        if self.policy_cfg["use_estimation"]:
            self.writer.add_scalar('Loss/privileged_loss', locs['mse_loss'], locs['it'])
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_cost', statistics.mean(locs['costbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_cost/time', statistics.mean(locs['costbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Regression loss:':>{pad}} {locs['mse_loss']:.4f}\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean cost:':>{pad}} {statistics.mean(locs['costbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        if self.policy_cfg["use_estimation"]:
            torch.save({
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'iter': self.current_learning_iteration,
                'infos': infos,
                'privi_estimator_state_dict': self.privileged_obs_estimator.state_dict(),
                # 'privi_optimizer_state_dict': self.privileged_obs_optimizer.state_dict()
                }, path)
            # torch.save(self.privileged_obs_estimator.model, path.replace('.pt', '_esti.pt'))
        else:
            torch.save({
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'iter': self.current_learning_iteration,
                'infos': infos
                }, path)
            

    def load(self, path, load_optimizer=True, load_estimator=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        loaded_dict_privi = loaded_dict
        if self.policy_cfg["use_estimation"] and load_estimator:
            self.privileged_obs_estimator.load_state_dict(loaded_dict_privi['privi_estimator_state_dict'])
            # if load_optimizer:
            #     self.privileged_obs_optimizer.load_state_dict(loaded_dict['privi_optimizer_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_inference_policy_and_encoder(self, device=None):
        self.alg.actor_critic.eval()
        self.alg.actor_critic.encoder.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference, self.alg.actor_critic.encoder, self.alg.actor_critic.decoder
    
class Estimator(nn.Module):
    def __init__(self, input_dim=50, history_length=50, output_dim=1, nn_type='mlp',device='cpu'):
        super(Estimator, self).__init__()
        self.input_dim = input_dim
        self.history_length = history_length
        self.output_dim = output_dim
        self.nn_type = nn_type
        # default: input=49, hist_len=50, output=1
        if nn_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(input_dim*history_length, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        elif nn_type == 'cnn':
            self.model = nn.Sequential(
                nn.Conv1d(self.input_dim, 16, kernel_size=7, stride=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=7, stride=3, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64*3, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )
        elif nn_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, 64, 2, batch_first=True)
            self.fc1 = nn.Linear(64, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, output_dim)
            self.model = nn.Sequential(
                self.lstm,
                self.fc1,
                self.relu,
                self.fc2
            )
        elif nn_type == 'gru':
            self.model = nn.GRU(input_dim, 64, 2, batch_first=True)
            self.fc1 = nn.Linear(64, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, output_dim)
        elif nn_type == 'mhmlp':
            self.model = nn.Sequential(
                nn.Linear(1+(input_dim-1)*history_length, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 64)
            )
            self.fc_mass = nn.Linear(64, 1)
            # self.fc_pushxy = nn.Linear(64, 2)
            self.fc_com = nn.Linear(64, 3)
            self.fc_friction = nn.Linear(64, 1)
            # self.fc_pushxy = nn.Linear(64, 2)
            # self.fc_kpkd = nn.Linear(64, 2)
        else:
            raise NotImplementedError(f"Estimator type {nn_type} not implemented")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.device = device
        self.to(device)

    def forward(self, x):
        
        if self.nn_type == 'lstm':  
            x, _ = self.model(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
        elif self.nn_type == 'gru':
            x, _ = self.model(x)
            x = x[:,-1,:]
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
        elif self.nn_type == 'cnn':
            # x = x.unsqueeze(1)
            x = x.transpose(1,2)
            x = self.model(x)
        elif self.nn_type == 'mlp':
            x = x.flatten(-2)
            x = self.model(x)
        elif self.nn_type == 'mhmlp':
            t = x[:,-1,13:14]
            x = torch.cat((x[:,:,0:13],x[:,:,14:50]),dim=-1)
            # td = torch.cat((g,t),dim=-1).squeeze(1)
            x = x.flatten(-2)
            x = torch.cat((x,t),dim=-1)
            x = self.model(x)
            x_mass = self.fc_mass(x)
            # x_pushxy = self.fc_pushxy(x)
            x_com = self.fc_com(x)
            x_friction = self.fc_friction(x)
            # x_pushxy = self.fc_pushxy(x)
            # x_kpkd = self.fc_kpkd(x)
            x = torch.cat((x_mass,x_com,x_friction),dim=-1)
        else:
            raise NotImplementedError(f"Estimator type {self.nn_type} not implemented")
        return x
