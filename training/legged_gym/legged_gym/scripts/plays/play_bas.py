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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import time
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time

EXPORT_POLICY = True
RECORD_FRAMES = True
MOVE_CAMERA = False
override = False # if True, the robot will keep moving by overriding the position target
mass_reset = True # if True, the mass will be reset to the initial value after each episode
def play(args, name):
    args.collect_data = False
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1280)
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.mesh_type = "plane"
    # env_cfg.terrain.terrain_types = ['flat']  # do not duplicate!

    # env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.friction_range = [0.5, 1.5]
    # env_cfg.domain_rand.push_robots = True
    # env_cfg.domain_rand.max_push_vel_xy = 1.0
    env_cfg.domain_rand.randomize_dof_bias = False
    env_cfg.domain_rand.erfi = False
    env_cfg.domain_rand.randomize_base_mass = True
    args.payload = 0.0 if args.payload is None else args.payload
    # env_cfg.domain_rand.added_mass_range = [args.payload, args.payload]
    env_cfg.domain_rand.added_mass_range = [3.0, 3.0]
    env_cfg.domain_rand.external_push_force = False
    env_cfg.domain_rand.randomize_kp_kd = False
    env_cfg.domain_rand.randomize_timer_minus = 0.0
    env_cfg.domain_rand.external_push_robots = False
    env_cfg.domain_rand.randomize_kp_kd = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_dof_bias = False
    env_cfg.domain_rand.erfi = False
    # env_cfg.domain_rand.com_pos_x_range = [-0.0, 0.0]
    # env_cfg.domain_rand.com_pos_y_range = [-0.0, 0.0]
    # env_cfg.domain_rand.com_pos_z_range = [-0.0, 0.0]
    print('testing {}-{}kg payload'.format(env_cfg.domain_rand.added_mass_range[0], env_cfg.domain_rand.added_mass_range[1]))
    
    # if args.load_run.count('exim') > 0:
    #     print('EXIM')
    #     env_cfg.env.privilege_enable = False
    #     env_cfg.env.num_observations = 61
    #     train_cfg.policy.decoder_enabled=True
    #     train_cfg.policy.use_estimation = False
    #     play_mode = 2
    # else:
    #     print('EX')
    #     play_mode = 1 
    #     env_cfg.env.privilege_enable = False
    #     env_cfg.env.num_observations = 61
    #     train_cfg.policy.decoder_enabled=False
    #     train_cfg.policy.use_estimation = True
   
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.debug_viz = True
    obs = env.get_observations()
    latent = torch.zeros((env_cfg.env.num_envs, train_cfg.policy.latent_dim+train_cfg.policy.privileged_dim), device=env.device)
    # if play_mode==2: obs = torch.cat((obs,latent),dim=-1)
    env.terrain_levels[:] = 9
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner.privileged_obs_estimator.eval()
    policy = ppo_runner.get_inference_policy(device=env.device)
    play_mode = 1
    # if play_mode==2:policy,encoder = ppo_runner.get_inference_policy_and_encoder(device=env.device)
    exported_policy_name = str(task_registry.loaded_policy_path.split('/')[-2]) + str(task_registry.loaded_policy_path.split('/')[-1])
    print('Loaded policy from: ', task_registry.loaded_policy_path)
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, exported_policy_name)
        print('Exported policy as jit script to: ', os.path.join(path, exported_policy_name))
        if play_mode==2: 
            torch.save(encoder, os.path.join(path.replace('policies','encoders'), str(task_registry.loaded_policy_path.split('/')[-2]) + '_encoder.pt'))

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 5 # which joint is used for logging
    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = 3*env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    obs_history = torch.zeros((env_cfg.env.num_envs, train_cfg.policy.history_length, 50), device=env.device)
    obs_history_dataset = []
    mass_dataset = []
    estimator_path = "default"
    if play_mode==1:
        # estimator_path = "legged_gym/logs/go1_pos_rough/abs_n_mass_co_train_dagger_stand/model_2200_privi.pt"
        # estimator_path = "legged_gym/logs/go1_pos_rough/abs_n_mass_co_train/model_10000_privi_extended.pt"
        # estimator_path = 'legged_gym/logs/go1_pr_rough/abs_n_mass_strict_dagger/model_13000_privi.pt'
        # estimator_path = 'legged_gym/logs/go1_pr_rough/abs_n_mass_strict_cotrain/model_16000_privi.pt'
        # estimator_path = "share/deploy/abs_n_mass_strict_mlp_pred_privi.pt"
        # estimator_path = "legged_gym/logs/go1_pr_rough/08_19_14-35-52_/model_14600_privi.pt" # noised
        # estimator_path = 'legged_gym/logs/go1_pos_rough/abs_n_mass_co_train_norao_dagger_stand/model_5000_privi.pt'
        # estimator_path = "legged_gym/logs/go1_pos_rough/abs_n_mcomf_norao_fus_esti/model_5000_privi.pt"
        # estimator_path = "legged_gym/logs/go1_pos_rough/abs_n_mcomp_esti/model_15000_privi.pt"
        # estimator_path = 'legged_gym/logs/go1_pos_rough/ex_mf_fus01/model_10000_privi.pt'

        
        # estimator = torch.load(estimator_path)
        estimator = ppo_runner.privileged_obs_estimator.model
        estimator.eval()
        # torch.save(estimator,
                #    os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', args.load_run+'_privi_whole.pt'))
        # torch.save(estimator.model, 
        #            os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', args.load_run+'_privi_model.pt'))
        # torch.save(estimator.fc_mass, 
        #            os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', args.load_run+'_privi_fc_mass.pt'))
        # torch.save(estimator.fc_com,
        #             os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', args.load_run+'_privi_fc_com.pt'))
        # torch.save(estimator.fc_friction,
        #             os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', args.load_run+'_privi_fc_friction.pt'))
        # torch.save(estimator.fc_pushxy,
        #             os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', args.load_run+'_privi_fc_pushxy.pt'))
        # torch.save(estimator.fc_kpkd,
        #             os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', args.load_run+'_privi_fc_kpkd.pt'))
    
    pred_mass_log =[]
    gt_mass_log = []
    pred_com_log = []
    gt_com_log = []
    pred_fric_log = []
    gt_fric_log = []
    mean_err_log = []
    privileged_dim = train_cfg.policy.privileged_dim
    latent_dim = train_cfg.policy.latent_dim
    estimated_pr = torch.zeros((env_cfg.env.num_envs, privileged_dim), device=env.device)

    pr=torch.zeros((env_cfg.env.num_envs, privileged_dim), device=env.device)
    for i in range(20*int(env.max_episode_length)):
        
        if play_mode==2:
            latent = encoder(obs_history)
            if i == 200:
                print('latent: ', torch.nn.functional.tanh(latent[0]))
            # reconstructed = decoder(latent[:,:train_cfg.policy.latent_dim])
            # rec_action = reconstructed[:,38:50]
            # actions = rec_action
            latent[:,:latent_dim] =torch.nn.functional.tanh(latent[:,:latent_dim])
            obs =torch.cat([obs,latent],dim=-1)
            # pr[:,2:]=pr[:,2:]*10
            estimated_pr=latent[:,latent_dim:].detach().clone()
            # estimated_pr[:,2:]=estimated_pr[:,2:]*10
            loss_explicit = torch.nn.functional.mse_loss(estimated_pr, pr)
            # print('explicit loss: ', loss_explicit)
            # print('gt: ', pr[0])
            # print('pred: ', latent[0,-privileged_dim:])

        obs = torch.cat((obs, estimated_pr), dim=-1)
        actions = policy(obs.detach())
        
        obs, pr, rews, dones, infos = env.step(actions.detach())
        # print(infos['episode']['rew_stand_still_action_pos'].cpu().numpy())
        # gt_com_log.append(infos['episode']['rew_stand_still_action_pos'].cpu().numpy())
        # print(torch.mean(torch.abs(obs[:,50:61]),dim=1))
        
        obs_history = torch.cat((obs_history[:,1:,:], obs[:,:50].unsqueeze(1)), dim=1)
        

        if args.collect_data:
            obs_history_dataset.append(obs_history.cpu().numpy())
            mass_dataset.append(pr.squeeze(-1).cpu().numpy())
            if i % 1000 == 0 and i>0:
                print(name,'-th dataset is Collecting data: ', str(i)+'/'+str(20*int(env.max_episode_length)))
                mass_dataset = np.array(mass_dataset)
                obs_history_dataset = np.array(obs_history_dataset)
                os.makedirs(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task,'exported', 'offline_dataset',args.load_run),exist_ok=True)
                np.savez(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task,'exported', 'offline_dataset',args.load_run,'data_'+name+'-'+str(i)+'.npz'), obs_histories=obs_history_dataset, added_masses=mass_dataset)
                mass_dataset = []
                obs_history_dataset = []
        if ppo_runner.policy_cfg['use_estimation'] and play_mode==1:
            with torch.no_grad():
                x = obs_history
                t = x[:,-1,13:14]
                x = torch.cat((x[:,:,0:13],x[:,:,14:50]),dim=-1)
                x = x.flatten(-2)
                x = torch.cat((x,t),dim=-1)
                estimated_pr_temp = estimator(x)
                # # if multi-head
                # x_mass = self.fc_mass(x)
                # # x_pushxy = self.fc_pushxy(x)
                # x_vel = self.fc_com(x)
                # x_friction = self.fc_friction(x)
                # x_pushxy = self.fc_pushxy(x)
                # x_kpkd = self.fc_kpkd(x)
                # x = torch.cat((x_mass,x_vel,x_friction,x_pushxy,x_kpkd),dim=-1)
                estimated_pr = torch.clamp(estimated_pr_temp, -5.0, 15.0)
                # mean_err = (torch.mean(torch.abs(estimated_pr-pr))).cpu().numpy()
                # mse_err = torch.mean(torch.sum((estimated_pr-pr)**2,dim=1),dim=0).cpu().numpy()
        
        with torch.no_grad():
            # Record estimation
            mass_err = torch.mean(torch.abs(estimated_pr[:,0]-pr[:,0])).cpu().numpy()
            # com_err = torch.mean(torch.abs(estimated_pr[:,1:4]-pr[:,1:4])).cpu().numpy()/10
            fric_err = torch.mean(torch.abs(estimated_pr[:,1]-pr[:,1])).cpu().numpy()
            # push_err = torch.mean(torch.abs(estimated_pr[:,2:]-pr[:,2:])).cpu().numpy()*10
            # kpkd_err = torch.mean(torch.abs(estimated_pr[:,7:9]-pr[:,7:9])).cpu().numpy()/10
            # mean_err = np.array([mass_err, push_err, vel_err, fric_err])
            if play_mode==1: obs = torch.cat((obs[:,:-train_cfg.policy.privileged_dim], estimated_pr), dim=-1) # using estimated added mass as policy input
            # obs = torch.cat((obs[:,:-1], 10.0*torch.ones_like(estimated_pr)), dim=-1) # using estimated added mass as policy input
            if i < stop_rew_log:
                # print('Estimated privilege: ', estimated_pr[0].cpu().numpy())
                # print('True privilege: ', pr[0].cpu().numpy())
                # print('step: ', i ,'mass_error: ', mass_err, 'push_error: ', push_err, 'friction_error: ', fric_err)
                # print('push_error: ', push_err, 'kpkd_error: ', kpkd_err)
                pass
            if i>50 and i < stop_rew_log:
                pred_mass_log.append(estimated_pr[1][0].flatten().cpu().numpy())
                gt_mass_log.append(pr[1][0].cpu().numpy())
                # pred_com_log.append(estimated_pr[0][1:4].flatten().cpu().numpy()/10.0)
                # gt_com_log.append(pr[0][1:4].cpu().numpy()/10.0)
                pred_fric_log.append(estimated_pr[0][-1].flatten().cpu().numpy())
                gt_fric_log.append(pr[0][-1].cpu().numpy())
                mean_err_log.append([mass_err,fric_err])
                # mean_err_log.append([mass_err, com_err, fric_err, push_err, kpkd_err])
        if i % (env.max_episode_length) == 0 and i > 0 and mass_reset:
            # obs_history[:] = 0
            # estimated_pr[:] = 0
            print('Resetting mass: step ', i)
            if env_cfg.domain_rand.randomize_base_mass:
            # self.envs_loads[env_ids] = self.envs_loads[env_ids].uniform_(self.cfg.domain_rand.min_base_mass, self.cfg.domain_rand.max_base_mass)
                for j in range(env_cfg.env.num_envs):
                    env_handle = env.envs[j]
                    actor_handle = env.actor_handles[j]
                    body_props = env.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
                    body_props = env._process_rigid_body_props(body_props, j)
                    env.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
                    # env.envs[j] = env_handle
                    # env.actor_handles[j] = actor_handle
        if override: 
            obs[:,10] = 0.0
            obs[:,11] = 0.0
            obs[:,12] = 0.0
            # env.timer_left[:] = 0.5 * env.max_episode_length_s
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            os.makedirs(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported','estimate',train_cfg.runner.load_run),exist_ok=True)
            logger.print_rewards()
            
if __name__ == '__main__':

    args = get_args()
    play(args, 0)
