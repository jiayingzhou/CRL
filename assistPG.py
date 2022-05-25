import copy
print('begin process')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
from collections import deque
import argparse
import time
from gym import wrappers
import os
parser = argparse.ArgumentParser()
parser.add_argument('--record', help='record', default=False)
parser.add_argument('--reward_threshold', help='reward', default=200, type=float)
parser.add_argument('--assist_round', help='assistance_round', default=2, type=int)
parser.add_argument('--iteration', help='iteration_round', default=999, type=int)
parser.add_argument('--device', help='GPU or CPU', default='cpu')
parser.add_argument('--env_run', help='environment', default='lunarlander')
parser.add_argument('--play_mode', help='oracle; single; assist; fl', default='assist')
parser.add_argument('--setting', help='choose test setting', default=1, type=int)
parser.add_argument('--mean_loss', help='The way choose loss: mean or not?', default=False)
parser.add_argument('--fl_round', help='rounds in FLSGD', default=50, type=int)
parser.add_argument('--fl_epoch', help='epochs for training in FLSGD', default=10, type=int)
parser.add_argument('--assist_epoch', help='epochs for training in SGD', default=10, type=int)
parser.add_argument('--episode', help='episode size in each epoch', default=8, type=int)
parser.add_argument('--epoch', help='epoche size', default=20, type=int)
parser.add_argument('--hid_size', help='hidden size', default=4, type=int)
parser.add_argument('--render_video', help='play video?', default=False)
parser.add_argument('--share_freq', help='frequency of share models', default=[4], type=int)
parser.add_argument('--converge_stop', help='define if need to stop when converge', default=False)
from gym.envs.box2d.lunar_landerV2 import LunarLander_new

from gym.envs.classic_control.cartpoleV2 import CartPoleEnv_new

args = parser.parse_args()
record = args.record
reward_threshold = args.reward_threshold
assistance_round = args.assist_round
iteration = args.iteration
# epoch_round = args.epoch_train_round
device = torch.device(args.device)
setting = args.setting
mean_loss = args.mean_loss
fl_epoch = args.fl_epoch
fl_round = args.fl_round
assist_epoch = args.assist_epoch
episode_size_each = args.episode
epoch_size_each = args.epoch
hid_size = args.hid_size
render_video = args.render_video
share_freq = args.share_freq
converge_stop = args.converge_stop
env_run = args.env_run
play_mode = args.play_mode

if env_run == 'lunarlander':
    environ = LunarLander_new()
if env_run == 'cartpole':
    environ = CartPoleEnv_new()

print('\n This experiment has no convergence stop between agents')
print('\n This experiment use fixed epoch for assistance')
print('\n The environment is ', env_run)
print('\n device is ', device)
print('\n record is ', record)
print('\n setting is ', setting)
print('\n iteration is', iteration)
print('\n The epoch size in assist is', assist_epoch)
print('\n The epoch size in FL is ', fl_epoch)
print('\n The maximum assist round is', assistance_round)
print('\n The maximum FL round is', fl_round)
print('\n The hidden size is ', hid_size)
print('\n The episode size in each epoch is', episode_size_each)
subfile = env_run +'/setting'+str(setting)+'/hid_size'+str(hid_size)+'/ite'+str(iteration)+'/'
if not os.path.exists('arl_video/'+subfile):
    os.makedirs('arl_video/'+subfile)
if not os.path.exists('result/'+subfile):
    os.makedirs('result/'+subfile)
if not os.path.exists('result/'+subfile+'share/'):
    os.makedirs('result/'+subfile+'share/')
if not os.path.exists('result/' + subfile + 'single_file/'):
    os.makedirs('result/' + subfile + 'single_file/')
if not os.path.exists('result/' + subfile + 'oracle_file/'):
    os.makedirs('result/' + subfile + 'oracle_file/')
if not os.path.exists('result/' + subfile + 'fl_file/'):
    os.makedirs('result/' + subfile + 'fl_file/')
if not os.path.exists('result/' + subfile + 'assist_file/'):
    os.makedirs('result/' + subfile + 'assist_file/')
if not os.path.exists('result/' + subfile + 'help_file/'):
    os.makedirs('result/' + subfile + 'help_file/')
if not os.path.exists('result/' + subfile + '_single_/'):
    os.makedirs('result/' + subfile + '_single_/')
if not os.path.exists('result/' + subfile + '_oracle_/'):
    os.makedirs('result/' + subfile + '_oracle_/')
if not os.path.exists('result/' + subfile + '_help_/'):
    os.makedirs('result/' + subfile + '_help_/')
def init_unif(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)
def init_const(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 0)
class Params:
    EPOCH_SIZE_IN_EACH_TRAJECTORY = epoch_size_each  # How many epochs we want to pack when transferring parameters    # How many parameters will be shared during each assistance
    ALPHA = 5e-3       # learning rate
    EPISODE_SIZE_IN_EACH_EPOCH = episode_size_each   # how many episodes we want to pack into an epoch
    GAMMA = 0.99        # discount rate
    HIDDEN_SIZE = hid_size    # number of hidden nodes we have in our dnn
    BETA = 0.1          # the entropy bonus multiplier
    reward_threshold = reward_threshold   # The stopping rule: when the last mean reward from 100 episodes > reward_threshold, stop!
    record = record   # whether to record the video or not

# Q-table is replaced by a neural network
class Agent(nn.Module):


    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        x = normalize(x, dim=1)
        x = self.model(x)
        return x


class PolicyGradient:
    def __init__(self, environment: str = "CartPole", map=None, msg1=None, msg2=None, share_frequncy=None):

        self.ALPHA = Params.ALPHA
        self.EPISODE_SIZE_IN_EACH_EPOCH = Params.EPISODE_SIZE_IN_EACH_EPOCH
        self.GAMMA = Params.GAMMA
        self.HIDDEN_SIZE = Params.HIDDEN_SIZE
        self.BETA = Params.BETA
        self.EPOCH_SIZE_IN_EACH_TRAJECTORY = Params.EPOCH_SIZE_IN_EACH_TRAJECTORY
        self.EPOCH_SIZE_IN_SHARE = share_frequncy
        self.reward_threshold = Params.reward_threshold
        # self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.DEVICE = device
        self.exit_flag = False
        self.msg1 = msg1
        self.msg2 = msg2
        self.record = record
        # create the environment
        self.env = environment
        # self.LEG_SPRING_TORQUE = LEG_SPRING_TORQUE
        # self.height_level = height_level
        self.map = map
        # self.map_n = map_n
        # self.epoch_round = epoch_round

        # the agent driven by a neural network architecture
        self.agent = Agent(observation_space_size=self.env.observation_space.shape[0],
                           action_space_size=self.env.action_space.n,
                           hidden_size=self.HIDDEN_SIZE).to(self.DEVICE)
        # if env_run == 'lunarlander':
        #     self.agent.apply(init_const)

        self.opt = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)
        # the total_rewards record mean rewards in the latest 100 episodes
        self.total_rewards_each_episode = deque([], maxlen=100)
        self.train_disc_reward_single_round = deque([], maxlen=100)
        # self.discount_rewards_each_epoch = []
        self.history_loss_each_epoch = deque([], maxlen=1000)
        self.train_discount_reward_each_epoch = deque([], maxlen=1000)   # record mean discounted reward in each epoch
        self.train_discount_reward_in_assistance = deque([], maxlen=assistance_round)  # record mean discounted reward in each assistance, only record the last epoch
        self.train_loss_in_assist = deque([], maxlen=assistance_round)
        # test
        self.test_rewards1 = deque([], maxlen=1000)    # record test addtive reward in each testing episode
        self.test_discount_reward1 = deque([], maxlen=1000) # record test discount reward in each testing episode
        self.test_seed1 = deque([], maxlen=1000)
        self.test_height1 = deque([], maxlen=1000)

        self.test_rewards2 = deque([], maxlen=1000)  # record test addtive reward in each testing episode
        self.test_discount_reward2 = deque([], maxlen=1000)  # record test discount reward in each testing episode
        self.test_seed2 = deque([], maxlen=1000)
        self.test_height2 = deque([], maxlen=1000)

        self.test_rewards3 = deque([], maxlen=1000)  # record test addtive reward in each testing episode
        self.test_discount_reward3 = deque([], maxlen=1000)  # record test discount reward in each testing episode
        self.test_seed3 = deque([], maxlen=1000)
        self.test_height3 = deque([], maxlen=1000)

        self.test_rewards4 = deque([], maxlen=1000)  # record test addtive reward in each testing episode
        self.test_discount_reward4 = deque([], maxlen=1000)  # record test discount reward in each testing episode
        self.test_seed4 = deque([], maxlen=1000)
        self.test_height4 = deque([], maxlen=1000)

        self.total_model_in_share = deque([], maxlen=1+int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY/self.EPOCH_SIZE_IN_SHARE))
        # This is for selecting parameters, different parameters in different models
        self.total_model_in_share.append(copy.deepcopy(self.agent.state_dict()))
        self.memory_epoch_mean = deque([], maxlen=1+int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY/self.EPOCH_SIZE_IN_SHARE))
        self.total_opt_in_share = deque([],
                                          maxlen=1 + int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY / self.EPOCH_SIZE_IN_SHARE))
        # flag to figure out if we have render a single episode current epoch
        self.finished_rendering_this_epoch = False
        # for federated learning
        self.test_rewards_fl = deque([], maxlen=fl_round)
        self.test_loss_fl = deque([], maxlen=fl_round)
        self.test_discount_rewards_fl = deque([], maxlen=fl_round)
        # self.train_rewards_fl = deque([], maxlen=fl_round)
        self.train_loss_fl = deque([], maxlen=fl_round)
        self.train_discount_rewards_fl = deque([], maxlen=fl_round)


    def test_on_environment(self, provider, test=True, fedrated=False, test_signal='t1'):
        # test_loss = []
        if self.record:
            provider.env = wrappers.Monitor(provider.env, './arl_video/' + subfile+ self.msg1 + '/' + self.msg2+'_on_' + provider.msg2, video_callable=lambda episode_id: True,
                                   force=True)
        # while epoch < self.EPOCH_SIZE_IN_EACH_TRAJECTORY:

        loss, mean_discount_rewards_epoch = provider.play_epoch_use_model_from_agent(self, test=test, test_signal=test_signal)


        # reset the rendering flag
        self.finished_rendering_this_epoch = False


            # test_loss.append(loss.data)
        if fedrated:
            self.test_loss_fl.append(loss)
            self.test_discount_rewards_fl.append(mean_discount_rewards_epoch)
        if self.record:
            self.env = self.env.unwrapped
        self.env.close()
        # return test_loss

    def get_assistance_from(self, provider, assist_round, test_agent1, test_agent2, test_agent3, test_agent4):
        """
                    The main interface for the Policy Gradient solver
                """

        # init the epoch arrays
        # used for entropy calculation
        # correct self.exit_flag first, to make sure both are converged
        self.exit_flag = False
        self.memory_epoch_mean = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY / self.EPOCH_SIZE_IN_SHARE))
        self.total_model_in_share = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY / self.EPOCH_SIZE_IN_SHARE))
        self.total_opt_in_share = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY / self.EPOCH_SIZE_IN_SHARE))

        # choose parameter who yield the smallest loss
        # We use all candidate parameters to perform episodes, and calculate loss (loss1 + loss2)
        # if len(provider.total_model_in_share) > share_length:
        #     length = share_length
        #     step_size = int(len(provider.total_model_in_share)/share_length)
        # else:
        #     length = len(provider.total_model_in_share)
        #     step_size = 1

        length = len(provider.total_model_in_share)
        step_size = 1

        # choose parameter from the provider first
        min_loss = np.Inf
        max_ind = 0
        for param_ind in range(length):
            param_cand = param_ind * step_size
            # provider.agent = provider.total_model_in_share[param_cand]
            self.agent.load_state_dict(copy.deepcopy(provider.total_model_in_share[param_cand]))
            self.opt.load_state_dict(copy.deepcopy(provider.total_opt_in_share[param_cand]))
            loss, discount_reward_epoch_mean = self.play_epoch_use_model_from_agent(self, record_reward=False)

            # take negative
            neg_discount_reward_epoch_mean = -(discount_reward_epoch_mean + provider.memory_epoch_mean[param_cand])
            if neg_discount_reward_epoch_mean < min_loss:
                max_ind = param_cand
                min_loss = neg_discount_reward_epoch_mean
                min_single_discount_reward = discount_reward_epoch_mean
                min_single_loss = loss
            # decide the parameters and use it as starting point
        # provider.agent = provider.total_model_in_share[max_ind]

        self.agent.load_state_dict(copy.deepcopy(provider.total_model_in_share[max_ind]))
        self.train_discount_reward_in_assistance.append(min_single_discount_reward)
        self.train_loss_in_assist.append(min_single_loss)
        if self.msg2 == 'u':
            # make test
            self.test_on_environment(test_agent1, test_signal='t1')
            self.test_on_environment(test_agent2, test_signal='t2')
            self.test_on_environment(test_agent3, test_signal='t3')
            self.test_on_environment(test_agent4, test_signal='t4')

            print('---------------------------------------------------------')
            print('\n ----------   Assistance ', assist_round, '  Feedback-------------------')
            print('\n The training loss at assistance', self.train_loss_in_assist)
            print('\n The training discount reward at this assistance',
                  self.train_discount_reward_in_assistance)

            print('\n main test reward mean on test 1: ')
            for iii in range(assist_round + 1):
                print(np.mean(self.test_rewards1[iii]), ',')
            print('\n main test reward mean on test 2: ')
            for iii in range(assist_round + 1):
                print(np.mean(self.test_rewards2[iii]), ',')
            print('\n main test reward mean on test 3: ')
            for iii in range(assist_round + 1):
                print(np.mean(self.test_rewards3[iii]), ',')
            print('\n main test reward mean on test 4: ')
            for iii in range(assist_round + 1):
                print(np.mean(self.test_rewards4[iii]), ',')

            print('\n ----------------------------------------------------------------')
            torch.save({
                'history_loss': self.history_loss_each_epoch,
                'model_state_dict': self.agent.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'test_rewards1': self.test_rewards1,
                'test_seed1': self.test_seed1,
                'test_height1': self.test_height1,
                'test_rewards2': self.test_rewards2,
                'test_seed2': self.test_seed2,
                'test_height2': self.test_height2,
                'test_rewards3': self.test_rewards3,
                'test_seed3': self.test_seed3,
                'test_height3': self.test_height3,
                'test_rewards4': self.test_rewards4,
                'test_seed4': self.test_seed4,
                'test_height4': self.test_height4,
                'test_discount_reward1': self.test_discount_reward1,
                'test_discount_reward2': self.test_discount_reward2,
                'test_discount_reward3': self.test_discount_reward3,
                'test_discount_reward4': self.test_discount_reward4,
                'episode_reward': self.total_rewards_each_episode,
                'discount_reward_in_assist': self.train_discount_reward_in_assistance,
                'u_share_loss': self.memory_epoch_mean,
                'u_share_model': self.total_model_in_share,
                'p_share_loss': provider.memory_epoch_mean,
                'p_share_model': provider.total_model_in_share,
            }, 'result/' + subfile + 'assist_file/assist' + str(assist_round))
        # init the episode and the epoch
        # init local training
        epoch = 0
        # loss_in_accumulate_epoch = deque([], maxlen=100)

        # launch a monitor at each round of assistance
        if self.record:
            self.env = wrappers.Monitor(self.env, './arl_video/' + subfile + self.msg1 + '/' + self.msg2 + '/' + str(
                assist_round), video_callable=lambda episode_id: True,
                                        force=True)
        expand = False
        while epoch < assist_epoch:
            loss, discount_reward_epoch_mean = self.play_epoch_use_model_from_agent(self, episode_expand=expand)

            if np.mean(self.total_rewards_each_episode) > 0.75 * self.reward_threshold:
                expand = True

            self.train_discount_reward_each_epoch.append(discount_reward_epoch_mean)

            # loss_in_accumulate_epoch.append(loss.data.cpu().numpy())
            # append loss and parameters for next round of assitance: user assist provider
            self.history_loss_each_epoch.append(loss.data)
            if epoch % self.EPOCH_SIZE_IN_SHARE == 1:
                self.memory_epoch_mean.append(discount_reward_epoch_mean)
                self.total_model_in_share.append(self.agent.state_dict())
                self.total_opt_in_share.append(self.opt.state_dict())
            # increment the epoch
            epoch += 1

            # reset the rendering flag
            self.finished_rendering_this_epoch = False
            # feedback
            print("\r",
                  f"Epoch: {epoch}, Avg Return rewards: {np.mean(self.total_rewards_each_episode):.3f}",
                  end="",
                  flush=True)

            # make epoch >=1, to make sure there is at least one gradient

            if np.mean(self.total_rewards_each_episode) > self.reward_threshold:
                # print('\n solved!!')
                # record the discount_reward_epoch_mean in the end of each assistance
                # self.train_discount_reward_in_assistance.append(discount_reward_epoch_mean)
                # self.train_loss_in_assist.append(loss.data.cpu().numpy())
                if converge_stop:
                    if epoch == 1:  # no train, already converge
                        self.exit_flag = True
                    break

            # zero the gradient
            self.opt.zero_grad()
            # backprop
            loss.backward()
            # update the parameters
            self.opt.step()

            # if epoch == (assist_epoch - 1):
                # record the discount_reward_epoch_mean in the end of each assistance
                # self.train_discount_reward_in_assistance.append(discount_reward_epoch_mean)
                # self.train_loss_in_assist.append(loss.data.cpu().numpy())
            if self.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

            del loss, discount_reward_epoch_mean

            # if np.mean(self.total_rewards_each_episode) > 100:
            #     self.opt = optim.SGD(params=self.agent.parameters(), lr=0.2*self.ALPHA)

        # unwrap the monitor for next round of wrapping
        if self.record:
            self.env = self.env.unwrapped
        # self.env.close()

    # close the environment
    #     self.env.close()

    # close the writer

    def self_train(self, tic, test_agent1=None, test_agent2=None, test_agent3=None,test_agent4=None, fedrated=False):
        """
                    The main interface for the Policy Gradient solver
                """

        # choose parameter who yield the smallest loss
        # We use all candidate parameters to perform episodes, and calculate loss (loss1 + loss2)

        self.memory_epoch_mean = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY / self.EPOCH_SIZE_IN_SHARE))
        self.total_model_in_share = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY / self.EPOCH_SIZE_IN_SHARE))
        self.total_opt_in_share = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY / self.EPOCH_SIZE_IN_SHARE))
        epoch = 0

        if self.record:
            self.env = wrappers.Monitor(self.env, './arl_video/' + subfile + self.msg1 + '/'+ self.msg2, video_callable=lambda episode_id: True,
                                   force=True)
        # First use episode_size
        expand = False
        while epoch < self.EPOCH_SIZE_IN_EACH_TRAJECTORY:
            loss, dicount_reward_epoch_mean = self.play_epoch_use_model_from_agent(self, episode_expand=expand)
            # add episode size if reward is large enough, not use on FL
            if not fedrated:
                if np.mean(self.total_rewards_each_episode) > 0.75 * self.reward_threshold:
                    expand = True

            if not fedrated:
                if epoch % (assist_epoch * 2) == 0:
                    self.test_on_environment(test_agent1, test_signal='t1')
                    self.test_on_environment(test_agent2, test_signal='t2')
                    self.test_on_environment(test_agent3, test_signal='t3')
                    self.test_on_environment(test_agent4, test_signal='t4')
                    self.train_disc_reward_single_round.append(dicount_reward_epoch_mean)
                    print('\n  test reward 1 mean at each round')
                    for jjj in range(len(self.test_rewards1)):
                        print('\n ', np.mean(self.test_rewards1[jjj]))
                    print('\n  test reward 2 mean at each round')
                    for jjj in range(len(self.test_rewards2)):
                        print('\n ', np.mean(self.test_rewards2[jjj]))
                    print('\n  test reward 3 mean at each round')
                    for jjj in range(len(self.test_rewards3)):
                        print('\n ', np.mean(self.test_rewards3[jjj]))
                    print('\n  test reward 4 mean at each round')
                    for jjj in range(len(self.test_rewards4)):
                        print('\n ', np.mean(self.test_rewards4[jjj]))
                    print('\n train disc reward')
                    print('\n ', self.train_disc_reward_single_round)

                    torch.save({
                        'model_state_dict': self.agent.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'train_disc_reward_round': self.train_disc_reward_single_round,
                        'test_rewards1': self.test_rewards1,
                        'test_rewards2': self.test_rewards2,
                        'test_rewards3': self.test_rewards3,
                        'test_rewards4': self.test_rewards4,

                    }, 'result/' + subfile + self.msg1 + '/' + str(epoch))




            # if the epoch is over - we have epoch trajectories to perform the policy gradient

            # reset the rendering flag
            self.finished_rendering_this_epoch = False
            # append loss and parameters for next round of assitance: user assist provider
            self.history_loss_each_epoch.append(loss.data)
            self.train_discount_reward_each_epoch.append(dicount_reward_epoch_mean)

            if epoch % self.EPOCH_SIZE_IN_SHARE == 0:
                self.memory_epoch_mean.append(dicount_reward_epoch_mean)
                self.total_model_in_share.append(self.agent.state_dict())
                self.total_opt_in_share.append(self.opt.state_dict())
                toc = time.time()
                print('\n The running minute is ', (toc - tic) / 60)
            # increment the epoch
            epoch += 1

            # feedback
            print("\r", f"Epoch: {epoch}, Avg Return rewards: {np.mean(self.total_rewards_each_episode):.3f}",
                  end="",
                  flush=True)


            if fedrated:
                if epoch == (self.EPOCH_SIZE_IN_EACH_TRAJECTORY):
                    # record the discount_reward_epoch_mean in the end of each fl
                    self.train_discount_rewards_fl.append(dicount_reward_epoch_mean)
                    self.train_loss_fl.append(loss)

            if self.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

            # zero the gradient
            self.opt.zero_grad()
            # backprop
            loss.backward()
            # update the parameters
            self.opt.step()

            del loss, dicount_reward_epoch_mean


        if self.record:
            self.env = self.env.unwrapped
    # close the environment
    #     self.env.close()

        # close the writer
        # self.writer.close()

    def play_episode_use_agent_lunar(self, provider, standardize=True, record_state=False, eng_p=None, seeds=None, engine_consumes=None):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """
        # reset the environment to a random initial state every epoch
        ll = len(self.map.h)
        index = torch.randperm(ll)[0]
        seed = self.map.seed[index]
        height = self.map.h[index]
        eng_consume = self.map.eng_consume[index]
        main_eng_power = self.map.main_eng_p[index]

        if record_state:
            main_eng_power = eng_p
            seed = seeds
            eng_consume = engine_consumes

        self.env.seed(int(seed))


        state = self.env.reset(height_level=height)

        if record_state:
            x = deque([], maxlen=1000)
            y = deque([], maxlen=1000)

        # initialize the episode arrays
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        cumulative_average_rewards = np.empty(shape=(0,), dtype=float)
        # cumulative_sd_rewards = np.empty(shape=(0,), dtype=float)
        episode_rewards = np.empty(shape=(0,), dtype=float)

        # episode loopf
        while True:

            # render the environment for the first episode in the epoch
            if render_video:
                if not self.finished_rendering_this_epoch:
                    self.env.render()

            # get the action logits from the agent - (preferences)
            action_logits = provider.agent(torch.tensor(state).float().unsqueeze(dim=0).to(self.DEVICE))

            # append the logits to the episode logits list
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # append the action to the episode action list to obtain the trajectory
            # we need to store the actions and logits so we could calculate the gradient of the performance
            episode_actions = torch.cat((episode_actions, action), dim=0)

            # take the chosen action, observe the reward and the next state

            if self.DEVICE.type == 'cpu':
                state, reward, done, _ = self.env.step(action=action.cpu().item(), MAIN_ENGINE_POWER=main_eng_power, eng_consume=eng_consume)
            else:
                state, reward, done, _ = self.env.step(action=action.cuda().item(), MAIN_ENGINE_POWER=main_eng_power, eng_consume=eng_consume)

            if record_state:
                # reocrd stateframe0.png
                x.append(state[0])
                y.append(state[1])
            # append the reward to the rewards pool that we collect during the episode
            # we need the rewards so we can calculate the weights for the policy gradient
            # and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

            # here the average reward is state specific
            cumulative_average_rewards = np.concatenate((cumulative_average_rewards,
                                              np.expand_dims(np.mean(episode_rewards), axis=0)),
                                             axis=0)
            # cumulative_sd_rewards = np.concatenate((cumulative_sd_rewards,
            #                                              np.expand_dims(np.std(episode_rewards), axis=0)),
            #                                             axis=0)

            if episode_logits.shape[0] > 500:   # done if the running steps are too much
                done = True

            episode_rewards_sum = sum(episode_rewards)
            if episode_rewards_sum < -250:
                done = True

            # the episode is over
            if done:

                # increment the episode


                # turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more rewards than the later taken actions
                discounted_rewards_to_go = PolicyGradient.get_discounted_rewards(rewards=episode_rewards,
                                                                                 gamma=self.GAMMA)
                discounted_tot_rewards = discounted_rewards_to_go[0]
                if standardize:
                    discounted_rewards_to_go -= cumulative_average_rewards  # baseline - state specific average
                    # discounted_tot_rewards /= cumulative_sd_rewards
                # # calculate the sum of the rewards for the running average metric
                sum_of_rewards = np.sum(episode_rewards)

                # set the mask for the actions taken in the episode
                mask = one_hot(episode_actions, num_classes=self.env.action_space.n)

                # calculate the log-probabilities of the taken actions
                # mask is needed to filter out log-probabilities of not related logits
                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

                # weight the episode log-probabilities by the rewards-to-go
                episode_weighted_log_probs = episode_log_probs * \
                    torch.tensor(discounted_rewards_to_go).float().to(self.DEVICE)

                # calculate the sum over trajectory of the weighted log-probabilities
                if not mean_loss:
                    sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)
                else:
                    sum_weighted_log_probs = torch.mean(episode_weighted_log_probs).unsqueeze(dim=0)

                # won't render again this epoch
                self.finished_rendering_this_epoch = True
                # env_wrap.close()
                self.env.close()

                if record_state:
                    return x, y, sum_of_rewards, seed
                else:
                    return sum_weighted_log_probs, episode_logits, sum_of_rewards,discounted_tot_rewards, seed, height


    def play_episode_use_agent_cart(self, provider, standardize=True, record_state=False, pole_l=None, seeds= None):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """
        # reset the environment to a random initial state every epoch
        ll = len(self.map.pole_len)
        index = torch.randperm(ll)[0]
        seed = self.map.seed[index]
        pole_len = self.map.pole_len[index]

        #
        #
        # state = self.env.reset(height_level=height, LEG_SPRING_TORQUE=torq)
        if record_state:
            pole_len = pole_l
            seed = seeds
        self.env.seed(int(seed))

        self.pole_length = pole_len
        state = self.env.reset()


        if record_state:
            x = deque([], maxlen=1000)
            y = deque([], maxlen=1000)

        # initialize the episode arrays
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        cumulative_average_rewards = np.empty(shape=(0,), dtype=float)
        # cumulative_sd_rewards = np.empty(shape=(0,), dtype=float)
        episode_rewards = np.empty(shape=(0,), dtype=float)

        # episode loopf
        while True:

            # render the environment for the first episode in the epoch
            if render_video:
                if not self.finished_rendering_this_epoch:
                    self.env.render(pole_len=pole_len)

            # get the action logits from the agent - (preferences)
            action_logits = provider.agent(torch.tensor(state).float().unsqueeze(dim=0).to(self.DEVICE))

            # append the logits to the episode logits list
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # append the action to the episode action list to obtain the trajectory
            # we need to store the actions and logits so we could calculate the gradient of the performance
            episode_actions = torch.cat((episode_actions, action), dim=0)

            # take the chosen action, observe the reward and the next state

            if self.DEVICE.type == 'cpu':
                state, reward, done, _ = self.env.step(action=action.cpu().item(),pole_len=pole_len)
            else:
                state, reward, done, _ = self.env.step(action=action.cuda().item(),pole_len=pole_len)
            if record_state:
                # reocrd stateframe0.png
                x.append(state[0])
                y.append(state[1])
            # append the reward to the rewards pool that we collect during the episode
            # we need the rewards so we can calculate the weights for the policy gradient
            # and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

            # here the average reward is state specific
            cumulative_average_rewards = np.concatenate((cumulative_average_rewards,
                                              np.expand_dims(np.mean(episode_rewards), axis=0)),
                                             axis=0)
            # cumulative_sd_rewards = np.concatenate((cumulative_sd_rewards,
            #                                              np.expand_dims(np.std(episode_rewards), axis=0)),
            #                                             axis=0)

            if episode_logits.shape[0] > 1000:   # done if the running steps are too much
                done = True

            episode_rewards_sum = sum(episode_rewards)
            if episode_rewards_sum < -250:
                done = True

            # the episode is over
            if done:

                # increment the episode


                # turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more rewards than the later taken actions
                discounted_rewards_to_go = PolicyGradient.get_discounted_rewards(rewards=episode_rewards,
                                                                                 gamma=self.GAMMA)
                discounted_tot_rewards = discounted_rewards_to_go[0]
                if standardize:
                    discounted_rewards_to_go -= cumulative_average_rewards  # baseline - state specific average
                    # discounted_tot_rewards /= cumulative_sd_rewards
                # # calculate the sum of the rewards for the running average metric
                sum_of_rewards = np.sum(episode_rewards)

                # set the mask for the actions taken in the episode
                mask = one_hot(episode_actions, num_classes=self.env.action_space.n)

                # calculate the log-probabilities of the taken actions
                # mask is needed to filter out log-probabilities of not related logits
                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

                # weight the episode log-probabilities by the rewards-to-go
                episode_weighted_log_probs = episode_log_probs * \
                    torch.tensor(discounted_rewards_to_go).float().to(self.DEVICE)

                # calculate the sum over trajectory of the weighted log-probabilities
                if not mean_loss:
                    sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)
                else:
                    sum_weighted_log_probs = torch.mean(episode_weighted_log_probs).unsqueeze(dim=0)

                # won't render again this epoch
                self.finished_rendering_this_epoch = True
                # env_wrap.close()
                self.env.close()

                if record_state:
                    return x, y, sum_of_rewards, seed

                else:
                    return sum_weighted_log_probs, episode_logits, sum_of_rewards,discounted_tot_rewards, seed, pole_len

    def calculate_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
            Calculates the policy "loss" and the entropy bonus
            Args:
                epoch_logits: logits of the policy network we have collected over the epoch
                weighted_log_probs: loP * W of the actions taken
            Returns:
                policy loss + the entropy bonus
                entropy: needed for logging
        """
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus

        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        if not mean_loss:
            entropy_bonus = -1 * self.BETA * entropy
        else:
            entropy_bonus = 0

        return policy_loss + entropy_bonus, entropy

    @staticmethod
    def get_discounted_rewards(rewards: np.array, gamma: float) -> np.array:
        """
            Calculates the sequence of discounted rewards-to-go.
            Args:
                rewards: the sequence of observed rewards
                gamma: the discount factor
            Returns:
                discounted_rewards: the sequence of the rewards-to-go
        """
        discounted_rewards = np.empty_like(rewards, dtype=float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
        return discounted_rewards

    def play_epoch_use_model_from_agent(self, provider, test=False, record_reward=True, test_signal=None, episode_expand=False): # record_reward = False, when receiving paras, no record


        epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)
        episode = 0
        discount_rewards_epoch = np.empty(shape=(0,), dtype=float)

        if test:
            episode_size = 2 * self.EPISODE_SIZE_IN_EACH_EPOCH
        else:
            episode_size = self.EPISODE_SIZE_IN_EACH_EPOCH
        if episode_expand:
            episode_size = 1 * self.EPISODE_SIZE_IN_EACH_EPOCH
        # discounted_tot_rewards_epoch = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)
        # sum_of_episode_rewards_epoch = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)
        discounted_tot_rewards_epoch = deque([], maxlen=episode_size)
        sum_of_episode_rewards_epoch = deque([], maxlen=episode_size)
        seed_epoch = deque([], maxlen=episode_size)
        height_epoch = deque([], maxlen=episode_size)
        while episode < episode_size:
            if env_run == 'lunarlander':
                (episode_weighted_log_prob_trajectory,
                 episode_logits,
                 sum_of_episode_rewards,
                 discounted_tot_rewards, seed, height
                 ) = self.play_episode_use_agent_lunar(provider)
            if env_run == 'cartpole':
                (episode_weighted_log_prob_trajectory,
                 episode_logits,
                 sum_of_episode_rewards,
                 discounted_tot_rewards, seed, height
                 ) = self.play_episode_use_agent_cart(provider)

            episode += 1

            # after each episode append the sum of total rewards to the deque
            if record_reward:
                if test:
                    # sum_of_episode_rewards_epoch = np.concatenate((sum_of_episode_rewards_epoch, np.array([sum_of_episode_rewards.cpu().numpy()])), axis=0)
                    # discounted_tot_rewards_epoch = np.concatenate((discounted_tot_rewards_epoch, np.array([discounted_tot_rewards.cpu().numpy()])), axis=0)
                    sum_of_episode_rewards_epoch.append(sum_of_episode_rewards)
                    discounted_tot_rewards_epoch.append(discounted_tot_rewards)
                    seed_epoch.append(seed)
                    height_epoch.append(height)
                else:
                    self.total_rewards_each_episode.append(sum_of_episode_rewards)


            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                                 dim=0)

            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)
            discount_rewards_epoch = np.concatenate((discount_rewards_epoch, np.array([discounted_tot_rewards])), axis=0)



        # reset the rendering flag
        self.finished_rendering_this_epoch = False
        if test:
            if test_signal == 't1':
                provider.test_rewards1.append(sum_of_episode_rewards_epoch)
                provider.test_discount_reward1.append(discounted_tot_rewards_epoch)
                provider.test_height1.append(height_epoch)
                provider.test_seed1.append(seed_epoch)
            if test_signal == 't2':
                provider.test_rewards2.append(sum_of_episode_rewards_epoch)
                provider.test_discount_reward2.append(discounted_tot_rewards_epoch)
                provider.test_height2.append(height_epoch)
                provider.test_seed2.append(seed_epoch)
            if test_signal == 't3':
                provider.test_rewards3.append(sum_of_episode_rewards_epoch)
                provider.test_discount_reward3.append(discounted_tot_rewards_epoch)
                provider.test_height3.append(height_epoch)
                provider.test_seed3.append(seed_epoch)
            if test_signal == 't4':
                provider.test_rewards4.append(sum_of_episode_rewards_epoch)
                provider.test_discount_reward4.append(discounted_tot_rewards_epoch)
                provider.test_height4.append(height_epoch)
                provider.test_seed4.append(seed_epoch)
        loss, _ = self.calculate_loss(epoch_logits=epoch_logits,
                                            weighted_log_probs=epoch_weighted_log_probs)
        # if the epoch is over - we have epoch trajectories to perform the policy gradient
        return loss, np.mean(discount_rewards_epoch)






def train_assist(user_map, provider_map, single_agent, test_agent1,test_agent2,test_agent3,test_agent4, message=None, share_frequncy=None):
    env = environ
    main_agent = PolicyGradient(environment=env,  map=user_map, msg1=message, msg2='u', share_frequncy=share_frequncy)
    assist_agent = PolicyGradient(environment=env,  map=provider_map, msg1=message, msg2='p', share_frequncy=share_frequncy)
    communication_round = assistance_round

    # deep copy the model from single agent
    main_agent.memory_epoch_mean = deque([], maxlen=int(main_agent.EPOCH_SIZE_IN_EACH_TRAJECTORY / main_agent.EPOCH_SIZE_IN_SHARE))
    main_agent.total_model_in_share = deque([], maxlen=int(main_agent.EPOCH_SIZE_IN_EACH_TRAJECTORY / main_agent.EPOCH_SIZE_IN_SHARE))

    # choose_index = int(assist_epoch/main_agent.EPOCH_SIZE_IN_SHARE)
    choose_index = min(assist_epoch, len(single_agent.total_model_in_share))
    for ind in range(0, choose_index, share_frequncy):
        print(ind)
        main_agent.total_model_in_share.append(copy.deepcopy(single_agent.total_model_in_share[ind]))
        main_agent.memory_epoch_mean.append(copy.deepcopy(single_agent.memory_epoch_mean[ind]))
        main_agent.total_opt_in_share.append(copy.deepcopy(single_agent.total_opt_in_share[ind]))

    for communication in range(communication_round):
        print("\n This is Assistance ", communication)

        tic = time.time()
        print("\n provider get assistance from user")
        assist_agent.get_assistance_from(main_agent, assist_round=communication, test_agent1=test_agent1,
                                         test_agent2=test_agent2, test_agent3=test_agent3, test_agent4=test_agent4)
        print('\n running time ', (time.time() - tic) / 60)

        print("\n user get assistance from provider")
        tic = time.time()
        main_agent.get_assistance_from(assist_agent, assist_round=communication, test_agent1=test_agent1,
                                       test_agent2=test_agent2, test_agent3=test_agent3, test_agent4=test_agent4)

        print('\n ----------------------------------------------------------------')
        if converge_stop:
            if main_agent.exit_flag:
                if assist_agent.exit_flag:
                    break
        print('\n running time ', (time.time() - tic) / 60)



    return main_agent, assist_agent

def single_train_and_test(single_agent, test_agent1,test_agent2, test_agent3,test_agent4, message=None,PRINT=True):
    # self train first
    tic = time.time()
    single_agent.self_train(tic=tic, test_agent1=test_agent1, test_agent2=test_agent2, test_agent3=test_agent3,test_agent4=test_agent4)
    single_agent.test_on_environment(test_agent1, test_signal='t1')
    single_agent.test_on_environment(test_agent2, test_signal='t2')
    single_agent.test_on_environment(test_agent3, test_signal='t3')
    single_agent.test_on_environment(test_agent4, test_signal='t4')
    single_agent.train_disc_reward_single_round.append(single_agent.train_discount_reward_each_epoch[-1])

    torch.save({
        'history_loss': single_agent.history_loss_each_epoch,
        'model_state_dict': single_agent.agent.state_dict(),
        'optimizer_state_dict': single_agent.opt.state_dict(),
        'epoch_disc_reward': single_agent.train_discount_reward_each_epoch,
        'train_disc_reward_': single_agent.train_disc_reward_single_round,
        'test_rewards1_round': single_agent.test_rewards1,
        'test_rewards2': single_agent.test_rewards2,
        'test_rewards3': single_agent.test_rewards3,
        'test_rewards4': single_agent.test_rewards4,
        'test_seed1': single_agent.test_seed1,
        'test_height1': single_agent.test_height1,
        'test_discount_reward1': single_agent.test_discount_reward1,
        'test_seed2': single_agent.test_seed2,
        'test_height2': single_agent.test_height2,
        'test_discount_reward2': single_agent.test_discount_reward2,
        'test_seed3': single_agent.test_seed3,
        'test_height3': single_agent.test_height3,
        'test_discount_reward3': single_agent.test_discount_reward3,
        'test_seed4': single_agent.test_seed4,
        'test_height4': single_agent.test_height4,
        'test_discount_reward4': single_agent.test_discount_reward4,
        'episode_reward': single_agent.total_rewards_each_episode,
    }, 'result/' + subfile + message)

    for ii in range(0, len(single_agent.total_model_in_share), 2*assist_epoch):
            torch.save({
                'u_share_loss': single_agent.memory_epoch_mean[ii],
                'u_share_model': single_agent.total_model_in_share[ii],
                'optimizer_state_dict': single_agent.total_opt_in_share[ii],
                'train_disc_reward_round': single_agent.train_disc_reward_single_round,
                'test_rewards1': single_agent.test_rewards1,
                'test_rewards2': single_agent.test_rewards2,
                'test_rewards3': single_agent.test_rewards3,
                'test_rewards4': single_agent.test_rewards4,
            }, 'result/' + subfile + message +'_file/' + str(ii))


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def train_fl(u_map, p_map, round, epoch_num, message, test_agent1, test_agent2, test_agent3, test_agent4, share_frequncy=None):

    main_agent = PolicyGradient(environment=env,  map=u_map, msg1=message, msg2='u', share_frequncy=share_frequncy)
    main_agent.EPOCH_SIZE_IN_EACH_TRAJECTORY = epoch_num
    assist_agent = PolicyGradient(environment=env, map=p_map, msg1=message, msg2='u', share_frequncy=share_frequncy)
    assist_agent.EPOCH_SIZE_IN_EACH_TRAJECTORY = epoch_num
    for rr in range(round):
        print('\n =================     FedSGD     ================')
        print('\n =========   This is round ', rr)
        global_w_list = []
        tic = time.time()
        main_agent.self_train(tic=tic, fedrated=True)
        if main_agent.exit_flag:
            break
        tic = time.time()
        assist_agent.self_train(tic=tic, fedrated=True)
        # average weights
        global_w_list.append(main_agent.agent.state_dict())
        global_w_list.append(assist_agent.agent.state_dict())
        global_weight = average_weights(global_w_list)
        main_agent.agent.load_state_dict(global_weight)
        assist_agent.agent.load_state_dict(global_weight)
        # test
        main_agent.test_on_environment(test_agent1, fedrated=True, test_signal='t1')
        main_agent.test_on_environment(test_agent2, fedrated=True, test_signal='t2')
        main_agent.test_on_environment(test_agent3, fedrated=True, test_signal='t3')
        main_agent.test_on_environment(test_agent4, fedrated=True, test_signal='t4')

        print('\n --------------   Round ', rr, '  performance   ---------------')


        print('\n The federated training loss', main_agent.train_loss_fl)
        print('\n The federated training discount reward', main_agent.train_discount_rewards_fl)

        print('\n main test reward mean on test 1: ')
        for iii in range(rr+1):
            print(np.mean(main_agent.test_rewards1[iii]), ',')
        print('\n main test reward mean on test 2: ')
        for iii in range(rr+1):
            print(np.mean(main_agent.test_rewards2[iii]), ',')
        print('\n main test reward mean on test 3: ')
        for iii in range(rr+1):
            print(np.mean(main_agent.test_rewards3[iii]), ',')
        print('\n main test reward mean on test 4: ')
        for iii in range(rr+1):
            print(np.mean(main_agent.test_rewards4[iii]), ',')

        print('\n --------------------------------------------------------------')


        torch.save({
            'fl_round': rr,
            'history_loss': main_agent.history_loss_each_epoch,
            'model_state_dict': main_agent.agent.state_dict(),
            'optimizer_state_dict': main_agent.opt.state_dict(),
            'epoch_disc_reward': main_agent.train_discount_reward_each_epoch,
            'train_discount_reward_fl': main_agent.train_discount_rewards_fl,
            'train_loss_fl': main_agent.train_loss_fl,
            'test_rewards1': main_agent.test_rewards1,
            'test_seed1': main_agent.test_seed1,
            'test_height1': main_agent.test_height1,
            'test_loss1': main_agent.test_discount_reward1,
            'test_rewards2': main_agent.test_rewards2,
            'test_seed2': main_agent.test_seed2,
            'test_height2': main_agent.test_height2,
            'test_loss2': main_agent.test_discount_reward2,
            'test_rewards3': main_agent.test_rewards3,
            'test_seed3': main_agent.test_seed3,
            'test_height3': main_agent.test_height3,
            'test_loss3': main_agent.test_discount_reward3,
            'test_rewards4': main_agent.test_rewards4,
            'test_seed4': main_agent.test_seed4,
            'test_height4': main_agent.test_height4,
            'test_loss4': main_agent.test_discount_reward4,
        }, 'result/' + subfile + 'fl_file/' + str(rr))


    return main_agent, assist_agent


def map_list_cart(map_u, map_p, map_t1, map_t2, map_t3, map_t4, ite):

    q1 = len(map_u.pole_len)

    q2 = len(map_p.pole_len)
    # map_all_steep = list(map_u.steep_level) + list(map_p.steep_level)
    # map_all_torq = list(map_u.torque) + list(map_p.torque)
    pole_len_all = list(map_u.pole_len) + list(map_p.pole_len)

    qt1 = len(map_t1.pole_len)

    qt2 = len(map_t2.pole_len)

    qt3 = len(map_t3.pole_len)

    qt4 = len(map_t4.pole_len)

    class u:
        # h = map_u.steep_level
        seed = list(np.arange(q1) + ite * (q1 + q2 + qt1 + qt2 + qt3 + qt4))
        # torq = list(map_u.torque)
        pole_len = list(map_u.pole_len)

    class p:
        # h = map_p.steep_level
        seed = list(np.arange(q1, q1 + q2) + ite * (q1 + q2 + qt1 + qt2 + qt3 + qt4))
        # torq = list(map_p.torque)
        pole_len = list(map_p.pole_len)

    class t1:
        # h = map_t1.steep_level
        seed = list(np.arange(q1 + q2, q1 + q2 + qt1) + ite * (q1 + q2 + qt1 + qt2 + qt3 + qt4))
        # torq = list(map_t1.torque)
        pole_len = list(map_t1.pole_len)

    class t2:
        # h = map_t2.steep_level
        seed = list(np.arange(q1 + q2 + qt1, q1 + q2 + qt1 + qt2) + ite * (q1 + q2 + qt1 + qt2 + qt3 + qt4))
        # torq = list(map_t2.torque)
        pole_len = list(map_t2.pole_len)

    class t3:
        # h = map_t3.steep_level
        seed = list(
            np.arange(q1 + q2 + qt1 + qt2, q1 + q2 + qt1 + qt2 + qt3) + ite * (q1 + q2 + qt1 + qt2 + qt3 + qt4))
        # torq = (map_t3.torque)
        pole_len = (map_t3.pole_len)

    class t4:
        # h = map_t4.steep_level
        seed = list(np.arange(q1 + q2 + qt1 + qt2 + qt3, q1 + q2 + qt1 + qt2 + qt3 + qt4) + ite * (
                    q1 + q2 + qt1 + qt2 + qt3 + qt4))
        # torq = list(map_t4.torque)
        pole_len = list(map_t4.pole_len)

    class o:
        # h = map_all_steep
        seed = list(np.arange(q1 + q2) + ite * (q1 + q2 + qt1 + qt2 + qt3 + qt4))
        # torq = list(map_all_torq)
        pole_len = list(pole_len_all)

    return u, p, o, t1, t2, t3, t4
def  map_list_lunar(map_u, map_p, map_t1,map_t2,map_t3,map_t4, ite):

    q1 = len(map_u.steep_level)

    q2 = len(map_p.steep_level)
    map_all_steep = list(map_u.steep_level) + list(map_p.steep_level)
    map_all_eng_consume = list(map_u.eng_consume) + list(map_p.eng_consume)
    map_all_eng = list(map_u.main_eng_power) + list(map_p.main_eng_power)

    qt1 = len(map_t1.steep_level)

    qt2 = len(map_t2.steep_level)

    qt3 = len(map_t3.steep_level)

    qt4 = len(map_t4.steep_level)


    class u:
        h = map_u.steep_level
        seed = list(np.arange(q1)+ite*(q1+q2+qt1+qt2+qt3+qt4))
        eng_consume = list(map_u.eng_consume)
        main_eng_p = list(map_u.main_eng_power)
    class p:
        h = map_p.steep_level
        seed = list(np.arange(q1, q1+q2)+ite*(q1+q2+qt1+qt2+qt3+qt4))
        eng_consume= list(map_p.eng_consume)
        main_eng_p = list(map_p.main_eng_power)
    class t1:
        h = map_t1.steep_level
        seed = list(np.arange(q1+q2, q1+q2+qt1)+ite*(q1+q2+qt1+qt2+qt3+qt4))
        eng_consume = list(map_t1.eng_consume)
        main_eng_p = list(map_t1.main_eng_power)
    class t2:
        h = map_t2.steep_level
        seed = list(np.arange(q1+q2+qt1, q1+q2+qt1+qt2)+ite*(q1+q2+qt1+qt2+qt3+qt4))
        eng_consume = list(map_t2.eng_consume)
        main_eng_p = list(map_t2.main_eng_power)
    class t3:
        h = map_t3.steep_level
        seed = list(np.arange(q1+q2+qt1+qt2, q1+q2+qt1+qt2+qt3)+ite*(q1+q2+qt1+qt2+qt3+qt4))
        eng_consume = (map_t3.eng_consume)
        main_eng_p = (map_t3.main_eng_power)
    class t4:
        h = map_t4.steep_level
        seed = list(np.arange(q1+q2+qt1+qt2+qt3, q1+q2+qt1+qt2+qt3+qt4)+ite*(q1+q2+qt1+qt2+qt3+qt4))
        eng_consume = list(map_t4.eng_consume)
        main_eng_p = list(map_t4.main_eng_power)
    class o:
        h = map_all_steep
        seed = list(np.arange(q1+q2)+ite*(q1+q2+qt1+qt2+qt3+qt4))
        eng_consume = list(map_all_eng_consume)
        main_eng_p = list(map_all_eng)
    return u, p, o, t1, t2, t3, t4


if __name__ == "__main__":
    if env_run == 'cartpole':
        if setting == 1:
            class map_user:
                pole_len = list(np.random.uniform(4, 5, 5))
            class map_provider:
                pole_len = list(np.random.uniform(0, 1, 5))
            class map_test1:
                pole_len = list(np.random.uniform(4, 5, 1000))
            class map_test2:
                pole_len = list(np.random.uniform(0, 5, 1000))
            class map_test3:
                pole_len = list(5 * np.random.beta(5, 1, 1000))
            class map_test4:
                pole_len = list(5 * np.random.beta(1, 5, 1000))
        u, p, o, t1, t2, t3, t4 = map_list_cart(map_user, map_provider, map_test1, map_test2, map_test3, map_test4,
                                                 iteration)
    if env_run == 'lunarlander':
        if setting == 1:
            class map_user:
                steep_level = [0.5] * 10
                eng_consume = [0.3] * 10
                main_eng_power = list(np.random.uniform(10, 15, 10))
            class map_provider:
                steep_level = [0.5] * 10
                eng_consume = [0.3] * 10
                main_eng_power = list(np.random.uniform(35, 40, 10))
            class map_test1:
                steep_level = [0.5] * 1000
                eng_consume = [0.3] * 1000
                main_eng_power = list(np.random.uniform(10, 15, 1000))
            class map_test2:
                steep_level = [0.5] * 1000
                eng_consume = [0.3] * 1000
                main_eng_power = list(np.random.uniform(10, 40, 1000))
            class map_test3:
                steep_level = [0.5] * 1000
                eng_consume = [0.3] * 1000
                main_eng_power = list(30 * np.random.beta(1, 5, 1000)+10)
            class map_test4:
                steep_level = [0.5] * 1000
                eng_consume = [0.3] * 1000
                main_eng_power = list(30 * np.random.beta(5, 1, 1000)+10)






        u, p, o, t1, t2, t3, t4 = map_list_lunar(map_user, map_provider, map_test1, map_test2, map_test3, map_test4,
                                           iteration)



    env = environ
    start = time.time()
    # test part
    test_agent1 = PolicyGradient(environment=env,
                                map=t1, msg1=None, msg2='t1', share_frequncy=1)
    test_agent2 = PolicyGradient(environment=env,
                                 map=t2, msg1=None, msg2='t2', share_frequncy=1)
    test_agent3 = PolicyGradient(environment=env,
                                 map=t3, msg1=None, msg2='t3', share_frequncy=1)
    test_agent4 = PolicyGradient(environment=env,
                                 map=t4, msg1=None, msg2='t4', share_frequncy=1)
    # train first

    if play_mode == 'assist':
        print('\n ============  AssistPG with share_freq=', share_freq[0], '================')
        help_agent = PolicyGradient(environment=env, map=u, msg1='_single_', msg2='u', share_frequncy=1)
        help_agent.EPOCH_SIZE_IN_EACH_TRAJECTORY = assist_epoch
        single_train_and_test(help_agent, test_agent1, test_agent2, test_agent3, test_agent4, message='help',
                              PRINT=True)
        main_agent1, assist_agent1 = train_assist(u, p, help_agent, test_agent1, test_agent2, test_agent3, test_agent4,
                                                  message='_assist_', share_frequncy=share_freq[0])
    if play_mode == 'single':

        print('\n ==============     Start trainig single agent  ==============')
        single_agent = PolicyGradient(environment=env, map=u, msg1='_single_', msg2='u', share_frequncy=1)
        single_train_and_test(single_agent, test_agent1, test_agent2, test_agent3, test_agent4, message='single',
                              PRINT=True)

    if play_mode == 'oracle':
        print('\n ==============     Start trainig oracle agent  ==============')
        oracle_agent = PolicyGradient(environment=env, map=o, msg1='_oracle_', msg2='u', share_frequncy=1)
        single_train_and_test(oracle_agent, test_agent1, test_agent2, test_agent3, test_agent4, message='oracle',
                              PRINT=True)
    if play_mode == 'fl':
        print('\n =================      Start FedPG     ================')

        user1, user2 = train_fl(u, p, round=fl_round, epoch_num=fl_epoch, message='_fl_', test_agent1=test_agent1,
                                test_agent2=test_agent2, test_agent3=test_agent3, test_agent4=test_agent4,
                                share_frequncy=1)
        ###########     PRINT     ##########################

        print('\n ==============        FedSGD      =======================')
        print('\n The federated testing loss', user1.test_loss_fl)
        print('\n The federated testing discount reward', user1.test_discount_rewards_fl)
        print('\n The federated training loss', user1.train_loss_fl)
        print('\n The federated training discount reward', user1.train_discount_rewards_fl)

