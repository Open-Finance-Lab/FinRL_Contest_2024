import os
import time
import torch.nn
import numpy as np
from torch import Tensor
from typing import Tuple, List

from erl_config import Config


class Evaluator:
    def __init__(self, cwd: str, env, args: Config):
        self.cwd = cwd  # current working directory to save model
        self.env = env  # the env for Evaluator, `eval_env = env` in default
        self.agent_id = args.gpu_id
        self.total_step = 0  # the total training step
        self.start_time = time.time()  # `used_time = time.time() - self.start_time`
        self.eval_times = args.eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = args.eval_per_step  # evaluate the agent per training steps
        self.eval_step_counter = -self.eval_per_step  # `self.total_step > self.eval_step_counter + self.eval_per_step`

        self.save_gap = args.save_gap
        self.save_counter = 0
        self.if_keep_save = args.if_keep_save
        self.if_over_write = args.if_over_write

        self.recorder_path = f'{cwd}/recorder.npy'
        self.recorder = []  # total_step, r_avg, r_std, obj_c, ...
        self.max_r = -np.inf
        print("| Evaluator:"
              "\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              "\n| `time`: Time spent from the start of training to this moment."
              "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `avgS`: Average of steps in an episode."
              "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'Time':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'objA':>7}{'etc.':>7}")

        self.prev_max_exp_r = -np.inf

    def evaluate_and_save(self, actor: torch.nn, steps: int, exp_r: float, logging_tuple: tuple):
        self.total_step += steps  # update total training steps
        if self.total_step < self.eval_step_counter + self.eval_per_step:
            return

        self.eval_step_counter = self.total_step

        rewards_step_ten = self.get_cumulative_rewards_and_step(actor)
        returns = rewards_step_ten[:, 0]  # episodic cumulative returns of an
        steps = rewards_step_ten[:, 1]  # episodic step number
        avg_r = returns.mean().item()
        std_r = returns.std().item()
        avg_s = steps.mean().item()
        std_s = steps.std().item()

        train_time = int(time.time() - self.start_time)
        self.recorder.append((self.total_step, avg_r, std_r, exp_r, *logging_tuple))  # update recorder

        '''print some information to Terminal'''
        prev_max_r = self.max_r
        self.max_r = max(self.max_r, avg_r)  # update max average cumulative rewards
        print(f"{self.agent_id:<3}{self.total_step:8.2e}{train_time:8.0f} |"
              f"{avg_r:8.2f}{std_r:7.1f}{avg_s:7.0f}{std_s:6.0f} |"
              f"{exp_r:8.2f}{''.join(f'{n:7.2f}' for n in logging_tuple)}")

        self.save_training_curve_jpg()

        if_save = avg_r > prev_max_r

        if not self.if_keep_save:
            return

        self.save_counter += 1
        actor_path = None
        if if_save:  # save checkpoint with the highest episode return
            if self.if_over_write:
                actor_path = f"{self.cwd}/actor.pth"
            else:
                actor_path = f"{self.cwd}/actor_{self.total_step:08}_{self.max_r:07.3f}.pth"

        elif self.save_counter >= self.save_gap:
            self.save_counter = 0
            if self.if_over_write:
                actor_path = f"{self.cwd}/actor.pth"
            else:
                actor_path = f"{self.cwd}/actor_{self.total_step:08}.pth"

        if actor_path:
            torch.save(actor, actor_path)  # save policy network in *.pt

    def save_or_load_recoder(self, if_save: bool):
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

    def get_cumulative_rewards_and_step(self, actor) -> Tensor:
        rewards_step_list = [get_cumulative_rewards_and_step_from_vec_env(self.env, actor)
                             for _ in range(max(1, self.eval_times // self.env.num_envs))]
        rewards_step_list = sum(rewards_step_list, [])
        rewards_step_ten = torch.tensor(rewards_step_list)
        return rewards_step_ten  # rewards_steps_ten.shape[1] == 2

    def save_training_curve_jpg(self):
        recorder = np.array(self.recorder)

        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        fig_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.max_r:.3f}"

        draw_learning_curve(recorder=recorder, fig_title=fig_title, save_path=f"{self.cwd}/LearningCurve.jpg")
        np.save(self.recorder_path, recorder)  # save self.recorder for `draw_learning_curve()`


"""util"""


def get_cumulative_rewards_and_steps(env, actor, if_render: bool = False) -> Tuple[float, int]:
    """Usage
    eval_times = 4
    net_dim = 2 ** 7
    actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pt'

    env = build_env(env_class=env_class, env_args=env_args)
    act = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))

    r_s_ary = [get_episode_return_and_step(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    """
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    for steps in range(max_step):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        if if_discrete:
            tensor_action = tensor_action.argmax(dim=1)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        returns += reward

        if if_render:
            env.render()
            time.sleep(0.02)

        if done:
            break
    else:
        print("| get_rewards_and_step: WARNING. max_step > 12345")
    returns = getattr(env, 'cumulative_returns', returns)
    steps += 1
    return returns, steps


def get_cumulative_rewards_and_step_from_vec_env(env, actor) -> List[Tuple[float, int]]:
    device = env.device
    env_num = env.num_envs
    max_step = env.max_step
    if_discrete = env.if_discrete

    '''get returns and dones (GPU)'''
    returns = torch.empty((max_step, env_num), dtype=torch.float32, device=device)
    dones = torch.empty((max_step, env_num), dtype=torch.bool, device=device)
    action_ints = []
    positions = []

    state = env.reset()  # must reset in vectorized env
    for t in range(max_step):
        action = actor(state.to(device))
        # assert action.shape == (env.env_num, env.action_dim)
        if if_discrete:
            action = action.argmax(dim=1, keepdim=True)
        state, reward, done, info_dict = env.step(action)

        returns[t] = reward
        dones[t] = done
        action_ints.append(env.action_int)
        positions.append(env.position)

    action_ary = torch.concatenate(action_ints, dim=0)
    action_ary = action_ary.float() # TODO for cpu only
    action_count = torch.histc(action_ary, bins=2 + 1, min=-1, max=1)
    action_count = action_count.data.cpu().numpy() / action_ary.shape[0]
    action_count = np.ceil(action_count * 998).astype(np.int32)

    position_ary = torch.concatenate(positions, dim=0)
    position_ary = position_ary.float()
    position_count = torch.histc(position_ary, bins=env.max_position * 2 + 1, min=-2, max=2)
    position_count = position_count.data.cpu().numpy() / position_ary.shape[0]
    position_count = np.ceil(position_count * 998).astype(np.int32)

    print(';;;;;;', ' ' * (67 + len('[333. 218. 450.] [  0. 340. 185. 475.   0.]')),
          action_count, position_count)

    '''get cumulative returns and step'''
    if hasattr(env, 'cumulative_returns'):  # GPU
        returns_step_list = [(ret, env.max_step) for ret in env.cumulative_returns]
    else:  # CPU
        returns = returns.cpu()
        dones = dones.cpu()

        returns_step_list = []
        for i in range(env_num):
            dones_where = torch.where(dones[:, i] == 1)[0] + 1
            episode_num = len(dones_where)
            if episode_num == 0:
                continue

            j0 = 0
            for j1 in dones_where.tolist():
                reward_sum = returns[j0:j1, i].sum().item()  # cumulative returns of an episode
                steps_num = j1 - j0  # step number of an episode
                returns_step_list.append((reward_sum, steps_num))

                j0 = j1
    return returns_step_list


def draw_learning_curve(recorder: np.ndarray = None,
                        fig_title: str = 'learning_curve',
                        save_path: str = 'learning_curve.jpg'):
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    obj_c = recorder[:, 4]
    obj_a = recorder[:, 5]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    '''axs[0]'''
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color01 = 'darkcyan'
    ax01.set_ylabel('Explore AvgReward', color=color01)
    ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    ax01.tick_params(axis='y', labelcolor=color01)

    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return', color=color0)
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std / 2, r_avg + r_std / 2, facecolor=color0, alpha=0.3)
    ax00.grid()
    '''axs[1]'''
    ax10 = axs[1]
    ax10.cla()

    ax11 = axs[1].twinx()
    color11 = 'darkcyan'
    ax11.set_ylabel('objC', color=color11)
    ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    ax11.tick_params(axis='y', labelcolor=color11)

    color10 = 'royalblue'
    ax10.set_xlabel('Total Steps')
    ax10.set_ylabel('objA', color=color10)
    ax10.plot(steps, obj_a, label='objA', color=color10)
    ax10.tick_params(axis='y', labelcolor=color10)
    for plot_i in range(6, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
    ax10.legend()
    ax10.grid()

    '''plot save'''
    plt.title(fig_title, y=2.3)
    plt.savefig(save_path)
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
