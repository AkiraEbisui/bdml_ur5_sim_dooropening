# Python
import copy
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

# Tensorflow
import tensorflow as tf

# CSV
import csv
import os
import sys

# ROS
import rospy
import rospkg

# import our training environment
import gym
from env.ur_door_opening_env import URSimDoorOpening

# import our training algorithms
from algorithm.ppo_gae import PPOGAEAgent

seed = rospy.get_param("/ML/seed")
obs_dim = rospy.get_param("/ML/obs_dim")
n_act = rospy.get_param("/ML/n_act")
epochs = rospy.get_param("/ML/epochs")
hdim = rospy.get_param("/ML/hdim")
policy_lr = rospy.get_param("/ML/policy_lr")
value_lr = rospy.get_param("/ML/value_lr")
max_std = rospy.get_param("/ML/max_std")
clip_range = rospy.get_param("/ML/clip_range")
n_step = rospy.get_param("/ML/n_step")
maxlen_num = rospy.get_param("/ML/maxlen_num")

gamma = rospy.get_param("/ML/gamma")
lam = rospy.get_param("/ML/lam")
episode_size = rospy.get_param("/ML/episode_size")
batch_size = rospy.get_param("/ML/batch_size")
nupdates = rospy.get_param("/ML/nupdates")
random_grasp = rospy.get_param("/ML/random_grasp")

agent = PPOGAEAgent(obs_dim, n_act, epochs, hdim, policy_lr, value_lr, max_std, clip_range, seed)
#agent = PPOGAEAgent(obs_dim, n_act, epochs=10, hdim=obs_dim, policy_lr=3e-3, value_lr=1e-3, max_std=1.0, clip_range=0.2, seed=seed)

'''
PPO Agent with Gaussian policy
'''
def run_episode(env, animate=False): # Run policy and collect (state, action, reward) pairs
    obs = env.reset()
    observes, actions, rewards, infos = [], [], [], []
    done = False

    for update in range(n_step):
        obs = np.array(obs)
        obs = obs.astype(np.float32).reshape((1, -1)) # numpy.ndarray (1, num_obs)
        observes.append(obs)
        
        action = agent.get_action(obs) # List
        actions.append(action)
        obs, reward, done, info = env.step(action, update)
        
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward) # List
        infos.append(info)

        if done is True:
            break
        
    return (np.concatenate(observes), np.array(actions), np.array(rewards, dtype=np.float32), infos)

def run_policy(env, episodes): # collect trajectories
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, infos = run_episode(env) # numpy.ndarray
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'infos': infos} 
        
#        print ("######################run_policy######################")
#        print ("observes: ", observes.shape, type(observes)) 		#(n_step, 21), <type 'numpy.ndarray'>
#        print ("actions: ", actions.shape, type(actions))  		#(n_step,  6), <type 'numpy.ndarray'>
#        print ("rewards: ", rewards.shape, type(rewards))  		#(n_step,   ), <type 'numpy.ndarray'>
#        print ("trajectory: ", len(trajectory), type(trajectory)) 	#(      ,  4), <type 'dict'>
#        print ("#####################run_policy#######################")
        
        trajectories.append(trajectory)
    return trajectories
        
def add_value(trajectories, val_func): # Add value estimation for each trajectories
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.get_value(observes)
        trajectory['values'] = values

def add_gae(trajectories, gamma, lam): # generalized advantage estimation (for training stability)
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        values = trajectory['values']
        
#        print ("###############################add_gae###########################")
#        print ("rewards: ", rewards.shape, type(rewards))  	# (n_step, ), <type 'numpy.ndarray'>
#        print ("values): ", values.shape, type(values))  	# (n_step, ), <type 'numpy.ndarray'>
#        print ("###############################add_gae###########################")

        # temporal differences        
        tds = rewards + np.append(values[1:], 0) * gamma - values
        advantages = np.zeros_like(tds)
        advantage = 0
        for t in reversed(range(len(tds))):
            advantage = tds[t] + lam * gamma * advantage
            advantages[t] = advantage
        trajectory['advantages'] = advantages

def add_rets(trajectories, gamma): # compute the value
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        
        returns = np.zeros_like(rewards)
        ret = 0
        for t in reversed(range(len(rewards))):
            ret = rewards[t] + gamma * ret
            returns[t] = ret            
        trajectory['returns'] = returns

def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    returns = np.concatenate([t['returns'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])

    # Normalization of advantages 
    # In baselines, which is a github repo including implementation of PPO coded by OpenAI, 
    # all policy gradient methods use advantage normalization trick as belows.
    # The insight under this trick is that it tries to move policy parameter towards locally maximum point.
    # Sometimes, this trick doesnot work.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, returns
    
def main():
    # Can check log msgs according to log_level {rospy.DEBUG, rospy.INFO, rospy.WARN, rospy.ERROR} 
    rospy.init_node('ur_gym', anonymous=True, log_level=rospy.DEBUG)

    os.mkdir('./results/' + sys.argv[1])
    
    env = gym.make('URSimDoorOpening-v0')
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed=seed)

    avg10_return_list = deque(maxlen=maxlen_num * 10) # 10
    avg_return_list = deque(maxlen=maxlen_num) # 10
    avg_knob_r_list = deque(maxlen=maxlen_num) # 10
    avg_panel_r_list = deque(maxlen=maxlen_num) # 10
    avg_action_r_list = deque(maxlen=maxlen_num) # 10
    avg_pol_loss_list = deque(maxlen=maxlen_num) # 10
    avg_val_loss_list = deque(maxlen=maxlen_num) # 10
    avg_entropy_list = deque(maxlen=maxlen_num) # 10
    avg_max_knob_rotation_list = deque(maxlen=maxlen_num) # 10
    avg_max_door_rotation_list = deque(maxlen=maxlen_num) # 10
    max_wrist3_list = deque(maxlen=maxlen_num) # 10
    min_wrist3_list = deque(maxlen=maxlen_num) # 10
    max_wrist2_list = deque(maxlen=maxlen_num) # 10
    min_wrist2_list = deque(maxlen=maxlen_num) # 10
    max_wrist1_list = deque(maxlen=maxlen_num) # 10
    min_wrist1_list = deque(maxlen=maxlen_num) # 10
    max_elb_list = deque(maxlen=maxlen_num) # 10
    min_elb_list = deque(maxlen=maxlen_num) # 10
    max_shl_list = deque(maxlen=maxlen_num) # 10
    min_shl_list = deque(maxlen=maxlen_num) # 10
    max_shp_list = deque(maxlen=maxlen_num) # 10
    min_shp_list = deque(maxlen=maxlen_num) # 10
    max_force_x_list = deque(maxlen=maxlen_num) # 10
    min_force_x_list = deque(maxlen=maxlen_num) # 10
    max_force_y_list = deque(maxlen=maxlen_num) # 10
    min_force_y_list = deque(maxlen=maxlen_num) # 10
    max_force_z_list = deque(maxlen=maxlen_num) # 10
    min_force_z_list = deque(maxlen=maxlen_num) # 10
    max_torque_x_list = deque(maxlen=maxlen_num) # 10
    min_torque_x_list = deque(maxlen=maxlen_num) # 10
    max_torque_y_list = deque(maxlen=maxlen_num) # 10
    min_torque_y_list = deque(maxlen=maxlen_num) # 10
    max_torque_z_list = deque(maxlen=maxlen_num) # 10
    min_torque_z_list = deque(maxlen=maxlen_num) # 10
    max_taxel0_list = deque(maxlen=maxlen_num) # 10
    min_taxel0_list = deque(maxlen=maxlen_num) # 10
    max_taxel1_list = deque(maxlen=maxlen_num) # 10
    min_taxel1_list = deque(maxlen=maxlen_num) # 10
    step_list = deque(maxlen=maxlen_num) # 10
    min_act_correct_list = deque(maxlen=maxlen_num) # 10
    max_act_correct_list = deque(maxlen=maxlen_num) # 10
    min_door_tolerance_list = deque(maxlen=maxlen_num) # 10
    max_door_tolerance_list = deque(maxlen=maxlen_num) # 10
    max_eef_x_list = deque(maxlen=maxlen_num) # 10
    min_eef_x_list = deque(maxlen=maxlen_num) # 10
    max_eef_y_list = deque(maxlen=maxlen_num) # 10
    min_eef_y_list = deque(maxlen=maxlen_num) # 10
    max_eef_z_list = deque(maxlen=maxlen_num) # 10
    min_eef_z_list = deque(maxlen=maxlen_num) # 10
    max_eef_rpy_x_list = deque(maxlen=maxlen_num) # 10
    min_eef_rpy_x_list = deque(maxlen=maxlen_num) # 10
    max_eef_rpy_y_list = deque(maxlen=maxlen_num) # 10
    min_eef_rpy_y_list = deque(maxlen=maxlen_num) # 10
    max_eef_rpy_z_list = deque(maxlen=maxlen_num) # 10
    min_eef_rpy_z_list = deque(maxlen=maxlen_num) # 10
    avg_success_x_hist_list = deque(maxlen=maxlen_num) # 10
    avg_fail_x_hist_list = deque(maxlen=maxlen_num) # 10
    avg_success_y_hist_list = deque(maxlen=maxlen_num) # 10
    avg_fail_y_hist_list = deque(maxlen=maxlen_num) # 10
    success_list = deque(maxlen=1) # 10
    success10_list = deque(maxlen=10) # 10
    success_rate_list = deque(maxlen=1) # 10

    # save fig
    x_data = []
    y_data = []
    y_data_knob_r = []
    y_data_panel_r = []
    y_data_action_r = []
    y_data_10return = []
    x_data_success_hist = []
    arr_x_data_success_hist = np.array([])
    x_data_fail_hist = []
    arr_x_data_fail_hist = np.array([])    
    y_data_success_hist = []
    arr_y_data_success_hist = np.array([])
    y_data_fail_hist = []
    arr_y_data_fail_hist = np.array([])
    y_data_success = []
    y_data_success_rate = []        
    x_data_v = []
    y_data_v = []
    x_data_p = []
    y_data_p = []
    x_data_e = []
    y_data_e = []
    x_data_k = []
    y_data_k = []
    x_data_d = []
    y_data_d = []
    x_data_a = []
    y_data_max_wrist3 = []
    y_data_min_wrist3 = []
    y_data_max_wrist2 = []
    y_data_min_wrist2 = []
    y_data_max_wrist1 = []
    y_data_min_wrist1 = []
    y_data_max_elb = []
    y_data_min_elb = []
    y_data_max_shl = []
    y_data_min_shl = []
    y_data_max_shp = []
    y_data_min_shp = []
    x_data_f = []
    y_data_max_force_x = []
    y_data_min_force_x = []
    y_data_max_force_y = []
    y_data_min_force_y = []
    y_data_max_force_z = []
    y_data_min_force_z = []
    y_data_max_torque_x = []
    y_data_min_torque_x = []
    y_data_max_torque_y = []
    y_data_min_torque_y = []
    y_data_max_torque_z = []
    y_data_min_torque_z = []
    y_data_min_taxel0 = []
    y_data_max_taxel0 = []
    y_data_min_taxel1 = []
    y_data_max_taxel1 = []
    y_data_step = []
    y_data_min_act_correct = []
    y_data_max_act_correct = []
    y_data_min_door_tolerance = []
    y_data_max_door_tolerance = []
    y_data_max_eef_x = []
    y_data_min_eef_x = []
    y_data_max_eef_y = []
    y_data_min_eef_y = []
    y_data_max_eef_z = []
    y_data_min_eef_z = []
    y_data_max_eef_rpy_x = []
    y_data_min_eef_rpy_x = []
    y_data_max_eef_rpy_y = []
    y_data_min_eef_rpy_y = []
    y_data_max_eef_rpy_z = []
    y_data_min_eef_rpy_z = []
    y_data_success = []
    y_data_success_rate = []

    fig = plt.figure(figsize=(20, 10))
    env.first_reset()

    ax1 = fig.add_subplot(4, 4, 1)
    ax1.plot(x_data, y_data, 'r-', label="rewards")
    ax1.plot(x_data, y_data_knob_r, 'b-', label="knob_rx30")
    ax1.plot(x_data, y_data_panel_r, 'g-', label="panel_r")
    ax1.plot(x_data, y_data_action_r, 'c-', label="action_rx30")
    ax1.set_xlabel("episodes")
    ax1.set_ylabel("ave_return")
    ax1.grid(axis='both')
    ax1.legend(loc=2)
    ax2 = fig.add_subplot(4, 4, 2)
    ax2.plot(x_data_v, y_data_v, 'b-', label="v_loss")
    ax2.set_xlabel("episodes")
    ax2.set_ylabel("ave_val_loss")
    ax2.grid(axis='both')
    ax2.legend(loc=2)
    ax3 = fig.add_subplot(4, 4, 3)
    ax3.plot(x_data_p, y_data_p, 'g-', label="p_loss")
    ax3.set_xlabel("episodes")
    ax3.set_ylabel("ave_pol_loss")
    ax3.grid(axis='both')
    ax3.legend(loc=2)
    ax4 = fig.add_subplot(4, 4, 4)
    ax4.plot(x_data_e, y_data_e, 'c-', label="entropy")
    ax4.set_xlabel("episodes")
    ax4.set_ylabel("entropy")
    ax4.grid(axis='both')
    ax4.legend(loc=3)
    ax5 = fig.add_subplot(4, 4, 5)
    ax5.plot(x_data_k, y_data_k, 'r-', label="knob")
    ax5.plot(x_data_k, y_data_d, 'b-', label="doorx10")
    ax5.set_xlabel("episodes")
    ax5.set_ylabel("max_knob&door_rotation")
    ax5.grid(axis='both')
    ax5.legend(loc=2)
    ax6 = fig.add_subplot(4, 4, 6)
    ax6.plot(x_data_a, y_data_max_wrist3, 'r-', linestyle="solid", label="w3")
    ax6.plot(x_data_a, y_data_min_wrist3, 'r-', linestyle="dashed")
    ax6.plot(x_data_a, y_data_max_wrist2, 'b-', linestyle="solid", label="w2")
    ax6.plot(x_data_a, y_data_min_wrist2, 'b-', linestyle="dashed")
    ax6.plot(x_data_a, y_data_max_wrist1, 'g-', linestyle="solid", label="w1")
    ax6.plot(x_data_a, y_data_min_wrist1, 'g-', linestyle="dashed")
    ax6.plot(x_data_a, y_data_max_elb, 'c-', linestyle="solid", label="el")
    ax6.plot(x_data_a, y_data_min_elb, 'c-', linestyle="dashed")
    ax6.plot(x_data_a, y_data_max_shl, 'm-', linestyle="solid", label="shl")
    ax6.plot(x_data_a, y_data_min_shl, 'm-', linestyle="dashed")
    ax6.plot(x_data_a, y_data_max_shp, 'k-', linestyle="solid", label="shp")
    ax6.plot(x_data_a, y_data_min_shp, 'k-', linestyle="dashed")
    ax6.set_xlabel("episodes")
    ax6.set_ylabel("max&min_action")
    ax6.grid(axis='both')
    ax6.legend(loc=2)
    ax7 = fig.add_subplot(4, 4, 7)
    ax7.plot(x_data_f, y_data_max_force_x, 'r-', linestyle="solid", label="fx")
    ax7.plot(x_data_f, y_data_min_force_x, 'r-', linestyle="dashed")
    ax7.plot(x_data_f, y_data_max_force_y, 'b-', linestyle="solid", label="fy")
    ax7.plot(x_data_f, y_data_min_force_y, 'b-', linestyle="dashed")
    ax7.plot(x_data_f, y_data_max_force_z, 'g-', linestyle="solid", label="fz")
    ax7.plot(x_data_f, y_data_min_force_z, 'g-', linestyle="dashed")
    ax7.set_xlabel("episodes")
    ax7.set_ylabel("max&min_force")
    ax7.grid(axis='both')
    ax7.legend(loc=3)
    ax8 = fig.add_subplot(4, 4, 8)
    ax8.plot(x_data_f, y_data_max_torque_x, 'r-', linestyle="solid", label="tqx")
    ax8.plot(x_data_f, y_data_min_torque_x, 'r-', linestyle="dashed")
    ax8.plot(x_data_f, y_data_max_torque_y, 'b-', linestyle="solid", label="tqy")
    ax8.plot(x_data_f, y_data_min_torque_y, 'b-', linestyle="dashed")
    ax8.plot(x_data_f, y_data_max_torque_z, 'g-', linestyle="solid", label="tqz")
    ax8.plot(x_data_f, y_data_min_torque_z, 'g-', linestyle="dashed")
    ax8.set_xlabel("episodes")
    ax8.set_ylabel("max&min_torque")
    ax8.grid(axis='both')
    ax8.legend(loc=3)
    ax9 = fig.add_subplot(4, 4, 9)
    ax9.plot(x_data_f, y_data_max_taxel0, 'r-', linestyle="solid", label="txl0")
    ax9.plot(x_data_f, y_data_min_taxel0, 'r-', linestyle="dashed")
    ax9.plot(x_data_f, y_data_max_taxel1, 'b-', linestyle="solid", label="txl1")
    ax9.plot(x_data_f, y_data_min_taxel1, 'b-', linestyle="dashed")
    ax9.set_xlabel("episodes")
    ax9.set_ylabel("min&max_taxel0&1")
    ax9.set_ylim(-0.7, 0.7)
    ax9.grid(axis='both')
    ax9.legend(loc=2)
    ax10 = fig.add_subplot(4, 4, 10)
    ax10.plot(x_data_f, y_data_step, 'r-', linestyle="solid", label="step")
    ax10.plot(x_data_f, y_data_max_act_correct, 'b-', linestyle="solid", label="act_correct")
    ax10.plot(x_data_f, y_data_min_act_correct, 'b-', linestyle="dashed")
    ax10.plot(x_data_f, y_data_max_door_tolerance, 'g-', linestyle="solid", label="door_tolerance")
    ax10.plot(x_data_f, y_data_min_door_tolerance, 'g-', linestyle="dashed")
    ax10.set_xlabel("episodes")
    ax10.set_ylabel("step&act_correct&door_tolerace")
    ax10.set_ylim(0, n_step)
    ax10.grid(axis='both')
    ax10.legend(loc=2)
    ax11 = fig.add_subplot(4, 4, 11)
    ax11.plot(x_data_f, y_data_max_eef_x, 'r-', linestyle="solid", label="eef_x")
    ax11.plot(x_data_f, y_data_min_eef_x, 'r-', linestyle="dashed")
    ax11.plot(x_data_f, y_data_max_eef_y, 'b-', linestyle="solid", label="eef_y")
    ax11.plot(x_data_f, y_data_min_eef_y, 'b-', linestyle="dashed")
    ax11.plot(x_data_f, y_data_max_eef_z, 'g-', linestyle="solid", label="eef_z")
    ax11.plot(x_data_f, y_data_min_eef_z, 'g-', linestyle="dashed")
    ax11.set_xlabel("episodes")
    ax11.set_ylabel("max&min_eef_xyz")
    ax11.grid(axis='both')
    ax11.legend(loc=3)
    ax12 = fig.add_subplot(4, 4, 12)
    ax12.plot(x_data_f, y_data_max_eef_rpy_x, 'r-', linestyle="solid", label="rpy_x")
    ax12.plot(x_data_f, y_data_min_eef_rpy_x, 'r-', linestyle="dashed")
    ax12.plot(x_data_f, y_data_max_eef_rpy_y, 'b-', linestyle="solid", label="rpy_y")
    ax12.plot(x_data_f, y_data_min_eef_rpy_y, 'b-', linestyle="dashed")
    ax12.plot(x_data_f, y_data_max_eef_rpy_z, 'g-', linestyle="solid", label="rpy_z")
    ax12.plot(x_data_f, y_data_min_eef_rpy_z, 'g-', linestyle="dashed")
    ax12.set_xlabel("episodes")
    ax12.set_ylabel("max&min_eef_rpy")
    ax12.grid(axis='both')
    ax12.legend(loc=3)
    ax13 = fig.add_subplot(4, 4, 13)
    ax13.plot(x_data, y_data_success, 'r-', linestyle="solid", label="success")
    ax13.set_xlabel("episodes")
    ax13.set_ylabel("success")
    ax13.grid(axis='both')
    ax13.legend(loc=3)
    ax14 = fig.add_subplot(4, 4, 14)
    ax14.hist(arr_x_data_success_hist, label="success")
    ax14.hist(arr_x_data_fail_hist, label="fail")
    ax14.set_xlabel("x")
    ax14.set_ylabel("success&fail")
    ax14.grid(axis='both')
    ax14.legend(loc=2)
    ax15 = fig.add_subplot(4, 4, 15)
    ax15.hist(arr_y_data_success_hist, label="success")
    ax15.hist(arr_y_data_fail_hist, label="fail")
    ax15.set_xlabel("y")
    ax15.set_ylabel("success&fail")
    ax15.grid(axis='both')
    ax15.legend(loc=2)
    ax16 = fig.add_subplot(4, 4, 16)
    ax16.plot(x_data, y_data_success_rate, 'r-', linestyle="solid")
    ax16.set_xlabel("episodes")
    ax16.set_ylabel("success_rate")
    ax16.grid(axis='both')

    for update in range(1, nupdates + 1):
        trajectories = run_policy(env, episodes=episode_size)                  # get trajectories(obs, act, reward, info)

        add_value(trajectories, agent)                                         # update value_nn and compute value (get v_t+1)
        add_gae(trajectories, gamma, lam)                                      # compute 1-1_GAE (add "advantages" to "trajectories")
        add_rets(trajectories, gamma)                                          # compute 2_value (add value to "trajectories" as "returns")
        observes, actions, advantages, returns = build_train_set(trajectories) # separate trajectories to data set like obs, act, advantages, and returns
        
#        print ("----------------------------------------------------")
#        print ("update: ", update)
#        print ("updates: ", nupdates)
#        print ("observes: ", observes.shape, type(observes)) 		# ('observes: ',   (n_step, 21), <type 'numpy.ndarray'>)
#        print ("advantages: ", advantages.shape, type(advantages))	# ('advantages: ', (n_step,),    <type 'numpy.ndarray'>)
#        print ("returns: ", returns.shape, type(returns)) 		# ('returns: ',    (n_step,),    <type 'numpy.ndarray'>)
#        print ("actions: ", actions.shape, type(actions)) 		# ('actions: ',    (n_step, 6),  <type 'numpy.ndarray'>)
#        print ("----------------------------------------------------")

        # compute 1_policy
        # compute 3_entropy

        pol_loss, val_loss, kl, entropy = agent.update(observes, actions, advantages, returns, batch_size=batch_size) # update agent and compute pol_loss, val_loss, kl, entropy

        success_x_hist = []
        fail_x_hist = []
        success_y_hist = []
        fail_y_hist = []

        if random_grasp >= 1:
            if env.success == 1:
                success_x_hist = env.rand_list[0] * 1000
#                print("success_x_hist", success_x_hist) # -1.3608912218786462
                arr_success_x_hist = np.array(success_x_hist)
#                print("arr_success_x_hist", arr_success_x_hist) # array(-1.36089122)
                avg_success_x_hist_list.append(arr_success_x_hist)
#                print("avg_success_x_hist_list", avg_success_x_hist_list) # deque([array(-0.97627008), array(1.24825577), array(-1.36089122)], maxlen=10)
                x_data_success_hist.append(avg_success_x_hist_list)
#                print("x_data_success_hist", x_data_success_hist) # deque([array(-0.97627008), array(1.24825577), array(-1.36089122)], maxlen=10), deque([array(-0.97627008), array(1.24825577), array(-1.36089122)], maxlen=10), deque([array(-0.97627008), array(1.24825577), array(-1.36089122)], maxlen=10)
                arr_x_data_success_hist = np.array(x_data_success_hist)
#                print("arr_x_data_success_hist", arr_x_data_success_hist) # array([[-0.97627008,  1.24825577, -1.36089122], [-0.97627008,  1.24825577, -1.36089122],[-0.97627008,  1.24825577, -1.36089122]])
                success_y_hist = env.rand_list[1] * 1000
                arr_success_y_hist = np.array(success_y_hist)
                avg_success_y_hist_list.append(arr_success_y_hist)
                y_data_success_hist.append(avg_success_y_hist_list)
                arr_y_data_success_hist = np.array(y_data_success_hist)        
            else:
                fail_x_hist = env.rand_list[0] * 1000
                arr_fail_x_hist = np.array(fail_x_hist)
                avg_fail_x_hist_list.append(arr_fail_x_hist)
                x_data_fail_hist.append(avg_fail_x_hist_list)
                arr_x_data_fail_hist = np.array(x_data_fail_hist)
                fail_y_hist = env.rand_list[1] * 1000
                arr_fail_y_hist = np.array(fail_y_hist)
                avg_fail_y_hist_list.append(arr_fail_y_hist)
                y_data_fail_hist.append(avg_fail_y_hist_list)
                arr_y_data_fail_hist = np.array(y_data_fail_hist) 

        success_times = env.success_times
        success_rate = float(0)

        if update < 10:
            success_rate = float(success_times) / update
        else:
            success_rate = (float(success_times) - success10_list[-9]) / 10

        avg_pol_loss_list.append(pol_loss)
        avg_val_loss_list.append(val_loss)
        avg_return_list.append([np.sum(t['rewards']) for t in trajectories])
        avg_knob_r_list.append(env.knob_rotation_r * 30)
        avg_panel_r_list.append(env.panel_rotation_r)
        avg_action_r_list.append(env.action_r * 30)
        avg_entropy_list.append(entropy)
        avg_max_knob_rotation_list.append(env.max_knob_rotation)
        avg_max_door_rotation_list.append(env.max_door_rotation * 10)
        max_wrist3_list.append(env.max_wrist3)
        min_wrist3_list.append(env.min_wrist3)
        max_wrist2_list.append(env.max_wrist2)
        min_wrist2_list.append(env.min_wrist2)
        max_wrist1_list.append(env.max_wrist1)
        min_wrist1_list.append(env.min_wrist1)
        max_elb_list.append(env.max_elb)
        min_elb_list.append(env.min_elb)
        max_shl_list.append(env.max_shl)
        min_shl_list.append(env.min_shl)
        max_shp_list.append(env.max_shp)
        min_shp_list.append(env.min_shp)
        max_force_x_list.append(env.max_force_x)
        min_force_x_list.append(env.min_force_x)
        max_force_y_list.append(env.max_force_y)
        min_force_y_list.append(env.min_force_y)
        max_force_z_list.append(env.max_force_z)
        min_force_z_list.append(env.min_force_z)
        max_torque_x_list.append(env.max_torque_x)
        min_torque_x_list.append(env.min_torque_x)
        max_torque_y_list.append(env.max_torque_y)
        min_torque_y_list.append(env.min_torque_y)
        max_torque_z_list.append(env.max_torque_z)
        min_torque_z_list.append(env.min_torque_z)
        min_taxel0_list.append(env.min_taxel0)
        max_taxel0_list.append(env.max_taxel0)
        min_taxel1_list.append(env.min_taxel1)
        max_taxel1_list.append(env.max_taxel1)
        step_list.append(returns.shape)
        min_act_correct_list.append(env.min_act_correct_n)
        max_act_correct_list.append(env.max_act_correct_n)
        min_door_tolerance_list.append(env.min_door_tolerance)
        max_door_tolerance_list.append(env.max_door_tolerance)
        max_eef_x_list.append(env.max_eef_x)
        min_eef_x_list.append(env.min_eef_x)
        max_eef_y_list.append(env.max_eef_y)
        min_eef_y_list.append(env.min_eef_y)
        max_eef_z_list.append(env.max_eef_z)
        min_eef_z_list.append(env.min_eef_z)
        max_eef_rpy_x_list.append(env.max_eef_rpy_x)
        min_eef_rpy_x_list.append(env.min_eef_rpy_x)
        max_eef_rpy_y_list.append(env.max_eef_rpy_y)
        min_eef_rpy_y_list.append(env.min_eef_rpy_y)
        max_eef_rpy_z_list.append(env.max_eef_rpy_z)
        min_eef_rpy_z_list.append(env.min_eef_rpy_z)
        success_list.append(env.success_times)
        success10_list.append(env.success_times)
        success_rate_list.append(float(success_rate))

        if (update%1) == 0:
            print('[{}/{}] success_rate : {}, n_step : {}, return : {:.3f}, value loss : {:.3f}, policy loss : {:.3f}, policy kl : {:.5f}, policy entropy : {:.3f}'.format(
                update, nupdates, success_rate, returns.shape, np.mean(avg_return_list), np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list), kl, entropy))

        x_data.append(update)
        y_data.append(np.mean(avg_return_list))
        y_data_knob_r.append(np.mean(avg_knob_r_list))
        y_data_panel_r.append(np.mean(avg_panel_r_list))
        y_data_action_r.append(np.mean(avg_action_r_list))
        x_data_v.append(update)
        y_data_v.append(np.mean(avg_val_loss_list))
        x_data_p.append(update)
        y_data_p.append(np.mean(avg_pol_loss_list))
        x_data_e.append(update)
        y_data_e.append(np.mean(avg_entropy_list))
        x_data_k.append(update)
        y_data_k.append(np.mean(avg_max_knob_rotation_list))
        x_data_d.append(update)
        y_data_d.append(np.mean(avg_max_door_rotation_list))
        x_data_a.append(update)
        y_data_max_wrist3.append(np.mean(max_wrist3_list))
        y_data_min_wrist3.append(np.mean(min_wrist3_list))
        y_data_max_wrist2.append(np.mean(max_wrist2_list))
        y_data_min_wrist2.append(np.mean(min_wrist2_list))
        y_data_max_wrist1.append(np.mean(max_wrist1_list))
        y_data_min_wrist1.append(np.mean(min_wrist1_list))
        y_data_max_elb.append(np.mean(max_elb_list))
        y_data_min_elb.append(np.mean(min_elb_list))
        y_data_max_shl.append(np.mean(max_shl_list))
        y_data_min_shl.append(np.mean(min_shl_list))
        y_data_max_shp.append(np.mean(max_shp_list))
        y_data_min_shp.append(np.mean(min_shp_list))
        x_data_f.append(update)
        y_data_max_force_x.append(np.mean(max_force_x_list))
        y_data_min_force_x.append(np.mean(min_force_x_list))
        y_data_max_force_y.append(np.mean(max_force_y_list))
        y_data_min_force_y.append(np.mean(min_force_y_list))
        y_data_max_force_z.append(np.mean(max_force_z_list))
        y_data_min_force_z.append(np.mean(min_force_z_list))
        y_data_max_torque_x.append(np.mean(max_torque_x_list))
        y_data_min_torque_x.append(np.mean(min_torque_x_list))
        y_data_max_torque_y.append(np.mean(max_torque_y_list))
        y_data_min_torque_y.append(np.mean(min_torque_y_list))
        y_data_max_torque_z.append(np.mean(max_torque_z_list))
        y_data_min_torque_z.append(np.mean(min_torque_z_list))
        y_data_min_taxel0.append(np.mean(min_taxel0_list))
        y_data_max_taxel0.append(np.mean(max_taxel0_list))
        y_data_min_taxel1.append(np.mean(min_taxel1_list))
        y_data_max_taxel1.append(np.mean(max_taxel1_list))
        y_data_step.append(np.mean(step_list))
        y_data_min_act_correct.append(np.mean(min_act_correct_list))
        y_data_max_act_correct.append(np.mean(max_act_correct_list))
        y_data_min_door_tolerance.append(np.mean(min_door_tolerance_list))
        y_data_max_door_tolerance.append(np.mean(max_door_tolerance_list))
        y_data_max_eef_x.append(np.mean(max_eef_x_list))
        y_data_min_eef_x.append(np.mean(min_eef_x_list))
        y_data_max_eef_y.append(np.mean(max_eef_y_list))
        y_data_min_eef_y.append(np.mean(min_eef_y_list))
        y_data_max_eef_z.append(np.mean(max_eef_z_list))
        y_data_min_eef_z.append(np.mean(min_eef_z_list))
        y_data_max_eef_rpy_x.append(np.mean(max_eef_rpy_x_list))
        y_data_min_eef_rpy_x.append(np.mean(min_eef_rpy_x_list))
        y_data_max_eef_rpy_y.append(np.mean(max_eef_rpy_y_list))
        y_data_min_eef_rpy_y.append(np.mean(min_eef_rpy_y_list))
        y_data_max_eef_rpy_z.append(np.mean(max_eef_rpy_z_list))
        y_data_min_eef_rpy_z.append(np.mean(min_eef_rpy_z_list))
        y_data_success.append(np.mean(success_list))
        y_data_success_rate.append(np.mean(success_rate_list))

        if (update%1) == 0:
            ax1.plot(x_data, y_data, 'r-')
            ax1.plot(x_data, y_data_knob_r, 'b-')
            ax1.plot(x_data, y_data_panel_r, 'g-')
            ax1.plot(x_data, y_data_action_r, 'c-')
            ax2.plot(x_data_v, y_data_v, 'b-')
            ax3.plot(x_data_p, y_data_p, 'g-',)
            ax4.plot(x_data_e, y_data_e, 'c-')
            ax5.plot(x_data_k, y_data_k, 'r-')
            ax5.plot(x_data_k, y_data_d, 'b-')
            ax6.plot(x_data_a, y_data_max_wrist3, 'r-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_wrist3, 'r-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_wrist2, 'b-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_wrist2, 'b-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_wrist1, 'g-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_wrist1, 'g-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_elb, 'c-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_elb, 'c-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_shl, 'm-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_shl, 'm-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_shp, 'k-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_shp, 'k-', linestyle="dashed")
            ax7.plot(x_data_f, y_data_max_force_x, 'r-', linestyle="solid")
            ax7.plot(x_data_f, y_data_min_force_x, 'r-', linestyle="dashed")
            ax7.plot(x_data_f, y_data_max_force_y, 'b-', linestyle="solid")
            ax7.plot(x_data_f, y_data_min_force_y, 'b-', linestyle="dashed")
            ax7.plot(x_data_f, y_data_max_force_z, 'g-', linestyle="solid")
            ax7.plot(x_data_f, y_data_min_force_z, 'g-', linestyle="dashed")
            ax8.plot(x_data_f, y_data_max_torque_x, 'r-', linestyle="solid")
            ax8.plot(x_data_f, y_data_min_torque_x, 'r-', linestyle="dashed")
            ax8.plot(x_data_f, y_data_max_torque_y, 'b-', linestyle="solid")
            ax8.plot(x_data_f, y_data_min_torque_y, 'b-', linestyle="dashed")
            ax8.plot(x_data_f, y_data_max_torque_z, 'g-', linestyle="solid")
            ax8.plot(x_data_f, y_data_min_torque_z, 'g-', linestyle="dashed")
            ax9.plot(x_data_f, y_data_max_taxel0, 'r-', linestyle="solid")
            ax9.plot(x_data_f, y_data_min_taxel0, 'r-', linestyle="dashed")
            ax9.plot(x_data_f, y_data_max_taxel1, 'b-', linestyle="solid")
            ax9.plot(x_data_f, y_data_min_taxel1, 'b-', linestyle="dashed")
            ax10.plot(x_data_f, y_data_step, 'r-', linestyle="solid")
            ax10.plot(x_data_f, y_data_max_act_correct, 'b-', linestyle="solid")
            ax10.plot(x_data_f, y_data_min_act_correct, 'b-', linestyle="dashed")
            ax10.plot(x_data_f, y_data_max_door_tolerance, 'g-', linestyle="solid")
            ax10.plot(x_data_f, y_data_min_door_tolerance, 'g-', linestyle="dashed")
            ax11.plot(x_data_f, y_data_max_eef_x, 'r-', linestyle="solid")
            ax11.plot(x_data_f, y_data_min_eef_x, 'r-', linestyle="dashed")
            ax11.plot(x_data_f, y_data_max_eef_y, 'b-', linestyle="solid")
            ax11.plot(x_data_f, y_data_min_eef_y, 'b-', linestyle="dashed")
            ax11.plot(x_data_f, y_data_max_eef_z, 'g-', linestyle="solid")
            ax11.plot(x_data_f, y_data_min_eef_z, 'g-', linestyle="dashed")
            ax12.plot(x_data_f, y_data_max_eef_rpy_x, 'r-', linestyle="solid")
            ax12.plot(x_data_f, y_data_min_eef_rpy_x, 'r-', linestyle="dashed")
            ax12.plot(x_data_f, y_data_max_eef_rpy_y, 'b-', linestyle="solid")
            ax12.plot(x_data_f, y_data_min_eef_rpy_y, 'b-', linestyle="dashed")
            ax12.plot(x_data_f, y_data_max_eef_rpy_z, 'g-', linestyle="solid")
            ax12.plot(x_data_f, y_data_min_eef_rpy_z, 'g-', linestyle="dashed")
            ax13.plot(x_data, y_data_success, 'r-', linestyle="solid", label="success")
            if random_grasp >= 1:
                if env.success == 1:
                    ax14.hist(arr_success_x_hist, facecolor='b') # array(-1.36089122)
                    ax15.hist(arr_success_y_hist, facecolor='b')
                else:
                    ax14.hist(arr_fail_x_hist, facecolor='r')
                    ax15.hist(arr_fail_y_hist, facecolor='r')
            ax16.plot(x_data, y_data_success_rate, 'r-', linestyle="solid")

            fig.subplots_adjust(hspace=0.3, wspace=0.4)
            plt.draw()
            plt.pause(1e-17)
            plt.savefig('./results/' + sys.argv[1] + '/ppo_with_gae_list.png')

            with open('results/' + sys.argv[1] + '/sample.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(x_data) #1
                writer.writerow(y_data) #2
                writer.writerow(y_data_v) #3
                writer.writerow(y_data_p) #4
                writer.writerow(y_data_e) #5
                writer.writerow(y_data_k) #6
                writer.writerow(y_data_d) #7
                writer.writerow(y_data_max_wrist3) #8
                writer.writerow(y_data_min_wrist3) #9
                writer.writerow(y_data_max_wrist2) #10
                writer.writerow(y_data_min_wrist2) #11
                writer.writerow(y_data_max_wrist1) #12
                writer.writerow(y_data_min_wrist1) #13
                writer.writerow(y_data_max_elb) #14
                writer.writerow(y_data_min_elb) #15
                writer.writerow(y_data_max_shl) #16
                writer.writerow(y_data_min_shl) #17
                writer.writerow(y_data_max_shp) #18
                writer.writerow(y_data_min_shp) #19
                writer.writerow(y_data_max_force_x) #20
                writer.writerow(y_data_min_force_x) #21
                writer.writerow(y_data_max_force_y) #22
                writer.writerow(y_data_min_force_y) #23
                writer.writerow(y_data_max_force_z) #24
                writer.writerow(y_data_min_force_z) #25
                writer.writerow(y_data_max_torque_x) #26
                writer.writerow(y_data_min_torque_x) #27
                writer.writerow(y_data_max_torque_y) #28
                writer.writerow(y_data_min_torque_y) #29
                writer.writerow(y_data_max_torque_z) #30
                writer.writerow(y_data_min_torque_z) #31
                writer.writerow(y_data_max_taxel0) #32
                writer.writerow(y_data_min_taxel0) #33
                writer.writerow(y_data_max_taxel1) #34
                writer.writerow(y_data_min_taxel1) #35
                writer.writerow(y_data_step) #36
                writer.writerow(y_data_max_act_correct) #37
                writer.writerow(y_data_min_act_correct) #38
                writer.writerow(y_data_max_door_tolerance) #39
                writer.writerow(y_data_min_door_tolerance) #40
                writer.writerow(y_data_max_eef_x) #41
                writer.writerow(y_data_min_eef_x) #42
                writer.writerow(y_data_max_eef_y) #43
                writer.writerow(y_data_min_eef_y) #44
                writer.writerow(y_data_max_eef_z) #45
                writer.writerow(y_data_min_eef_z) #46
                writer.writerow(y_data_max_eef_rpy_x) #47
                writer.writerow(y_data_min_eef_rpy_x) #48
                writer.writerow(y_data_max_eef_rpy_y) #49
                writer.writerow(y_data_min_eef_rpy_y) #50
                writer.writerow(y_data_max_eef_rpy_z) #51
                writer.writerow(y_data_min_eef_rpy_z) #52
                writer.writerow(avg_success_x_hist_list) #53
                writer.writerow(avg_success_y_hist_list) #54
                writer.writerow(avg_fail_x_hist_list) #55
                writer.writerow(avg_fail_y_hist_list) #56
                writer.writerow(y_data_success) #57
                writer.writerow(y_data_success_rate) #58

        if (np.mean(avg_max_knob_rotation_list) > 0.8 and np.mean(avg_max_door_rotation_list) > 7): # Threshold return to success 
            print('[{}/{}] return : {:.3f}, value loss : {:.3f}, policy loss : {:.3f}'.format(update,nupdates, np.mean(avg_return_list), np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list)))
            print('The problem is solved with {} episodes'.format(update*episode_size))
            break

    env.end_reset()
        #env.close() # rospy.wait_for_service('/pause_physics') -> raise ROSInterruptException("rospy shutdown")

if __name__ == '__main__':
    main()
