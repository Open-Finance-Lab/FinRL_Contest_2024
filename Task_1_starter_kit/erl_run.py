import os
import time
import torch
import numpy as np
from multiprocessing import Process, Pipe

from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator
from trade_simulator import TradeSimulator, EvalTradeSimulator

#
if os.name == "nt":  # if is WindowOS (Windows NT)
    """Fix bug about Anaconda in WindowOS
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""run"""


class Learner(Process):
    def __init__(
        self,
        learner_pipe: Pipe,
        worker_pipes: [Pipe],
        evaluator_pipe: Pipe,
        args: Config,
    ):
        super().__init__()
        self.recv_pipe = learner_pipe[0]
        self.send_pipes = [worker_pipe[1] for worker_pipe in worker_pipes]
        self.eval_pipe = evaluator_pipe[1]
        self.args = args

    def run(self):
        args = self.args
        torch.set_grad_enabled(False)

        """init agent"""
        agent = args.agent_class(
            args.net_dims,
            args.state_dim,
            args.action_dim,
            gpu_id=args.gpu_id,
            args=args,
        )
        agent.save_or_load_agent(args.cwd, if_save=False)

        """init buffer"""
        if args.if_off_policy:
            buffer = ReplayBuffer(
                gpu_id=args.gpu_id,
                num_seqs=args.num_envs * args.num_workers,
                max_size=args.buffer_size,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
            )
        else:
            buffer = []

        """loop"""
        if_off_policy = args.if_off_policy
        if_save_buffer = args.if_save_buffer
        num_workers = args.num_workers
        num_envs = args.num_envs
        state_dim = args.state_dim
        action_dim = args.action_dim
        horizon_len = args.horizon_len
        num_seqs = args.num_envs * args.num_workers
        num_steps = args.horizon_len * args.num_workers
        cwd = args.cwd
        del args

        agent.last_state = torch.empty(
            (num_seqs, state_dim), dtype=torch.float32, device=agent.device
        )

        states = torch.empty(
            (horizon_len, num_seqs, state_dim), dtype=torch.float32, device=agent.device
        )
        actions = torch.empty(
            (horizon_len, num_seqs, action_dim),
            dtype=torch.float32,
            device=agent.device,
        )
        rewards = torch.empty(
            (horizon_len, num_seqs), dtype=torch.float32, device=agent.device
        )
        undones = torch.empty(
            (horizon_len, num_seqs), dtype=torch.bool, device=agent.device
        )
        if if_off_policy:
            buffer_items_tensor = (states, actions, rewards, undones)
        else:
            logprobs = torch.empty(
                (horizon_len, num_seqs), dtype=torch.float32, device=agent.device
            )
            buffer_items_tensor = (states, actions, logprobs, rewards, undones)

        if_train = True
        while if_train:
            """Learner send actor to Workers"""
            for send_pipe in self.send_pipes:
                send_pipe.send(agent.act)

            """Learner receive (buffer_items, last_state) from Workers"""
            for _ in range(num_workers):
                worker_id, buffer_items, last_state = self.recv_pipe.recv()

                buf_i = worker_id * num_envs
                buf_j = worker_id * num_envs + num_envs
                for buffer_item, buffer_tensor in zip(
                    buffer_items, buffer_items_tensor
                ):
                    buffer_tensor[:, buf_i:buf_j] = buffer_item
                agent.last_state[buf_i:buf_j] = last_state

            """Learner update training data to (buffer, agent)"""
            if if_off_policy:
                buffer.update(buffer_items_tensor)
            else:
                buffer[:] = buffer_items_tensor

            """agent update network using training data"""
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            """Learner receive training signal from Evaluator"""
            if (
                self.eval_pipe.poll()
            ):  # whether there is any data available to be read of this pipe
                if_train = (
                    self.eval_pipe.recv()
                )  # True means evaluator in idle moments.
                actor = (
                    agent.act
                )  # so Leaner send an actor to evaluator for evaluation.
            else:
                actor = None

            """Learner send actor and training log to Evaluator"""
            exp_r = (
                buffer_items_tensor[2].mean().item()
            )  # the average rewards of exploration
            self.eval_pipe.send((actor, num_steps, exp_r, logging_tuple))

        """Learner send the terminal signal to workers after break the loop"""
        for send_pipe in self.send_pipes:
            send_pipe.send(None)

        """save"""
        agent.save_or_load_agent(cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, "save_or_load_history"):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)
            print(f"| LearnerPipe.run: ReplayBuffer saved  in {cwd}")


class Worker(Process):
    def __init__(
        self, worker_pipe: Pipe, learner_pipe: Pipe, worker_id: int, args: Config
    ):
        super().__init__()
        self.recv_pipe = worker_pipe[0]
        self.send_pipe = learner_pipe[1]
        self.worker_id = worker_id
        self.args = args

    def run(self):
        args = self.args
        worker_id = self.worker_id
        torch.set_grad_enabled(False)

        """init environment"""
        env = build_env(args.env_class, args.env_args, args.gpu_id)

        """init agent"""
        agent = args.agent_class(
            args.net_dims,
            args.state_dim,
            args.action_dim,
            gpu_id=args.gpu_id,
            args=args,
        )
        agent.save_or_load_agent(args.cwd, if_save=False)

        """init agent.last_state"""
        state = env.reset()
        if args.num_envs == 1:
            assert state.shape == (args.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(
                state, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
        else:
            assert state.shape == (args.num_envs, args.state_dim)
            assert isinstance(state, torch.Tensor)
            state = state.to(agent.device)
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        agent.last_state = state.detach()

        """init buffer"""
        horizon_len = args.horizon_len
        if args.if_off_policy:
            buffer_items = agent.explore_env(env, args.horizon_len, if_random=True)
            self.send_pipe.send((worker_id, buffer_items, agent.last_state))

        """loop"""
        del args

        while True:
            """Worker receive actor from Learner"""
            actor = self.recv_pipe.recv()
            if actor is None:
                break

            """Worker send the training data to Learner"""
            agent.act = actor
            buffer_items = agent.explore_env(env, horizon_len)
            self.send_pipe.send((worker_id, buffer_items, agent.last_state))

        env.close() if hasattr(env, "close") else None


class EvaluatorProc(Process):
    def __init__(self, evaluator_pipe: Pipe, args: Config):
        super().__init__()
        self.pipe = evaluator_pipe[0]
        self.args = args

    def run(self):
        args = self.args
        torch.set_grad_enabled(False)

        """wandb(weights & biases): Track and visualize all the pieces of your machine learning pipeline."""
        # wandb = None
        # if getattr(args, 'if_use_wandb', False):
        #     import wandb
        #     wandb_project_name = "train"
        #     wandb.init(project=wandb_project_name)

        """init evaluator"""
        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args)

        """loop"""
        cwd = args.cwd
        break_step = args.break_step
        device = torch.device(
            f"cuda:{args.gpu_id}"
            if (torch.cuda.is_available() and (args.gpu_id >= 0))
            else "cpu"
        )
        del args

        if_train = True
        while if_train:
            """Evaluator receive training log from Learner"""
            actor, steps, exp_r, logging_tuple = self.pipe.recv()
            # wandb.log({"obj_cri": logging_tuple[0], "obj_act": logging_tuple[1]}) if wandb else None

            """Evaluator evaluate the actor and save the training log"""
            if actor is None:
                evaluator.total_step += (
                    steps  # update total_step but don't update recorder
                )
            else:
                actor = actor.to(device)
                evaluator.evaluate_and_save(actor, steps, exp_r, logging_tuple)

            """Evaluator send the training signal to Learner"""
            if_train = (evaluator.total_step <= break_step) and (
                not os.path.exists(f"{cwd}/stop")
            )
            self.pipe.send(if_train)

        """Evaluator save the training log and draw the learning curve"""
        evaluator.save_training_curve_jpg()
        print(
            f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}"
        )

        eval_env.close() if hasattr(eval_env, "close") else None


def train_agent(args: Config):
    args.init_before_training()
    torch.set_grad_enabled(False)

    """init environment"""
    env = build_env(args.env_class, args.env_args, args.gpu_id)

    """init agent"""
    agent = args.agent_class(
        args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args
    )
    agent.save_or_load_agent(args.cwd, if_save=False)

    """init agent.last_state"""
    state = env.reset()
    if args.num_envs == 1:
        assert state.shape == (args.state_dim,)
        assert isinstance(state, np.ndarray)
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(
            0
        )
    else:
        if state.shape != (args.num_envs, args.state_dim):
            raise ValueError(
                f"state.shape == (num_envs, state_dim): {state.shape, args.num_envs, args.state_dim}"
            )
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"isinstance(state, torch.Tensor): {repr(state)}")
        state = state.to(agent.device)
    assert state.shape == (args.num_envs, args.state_dim)
    assert isinstance(state, torch.Tensor)
    agent.last_state = state.detach()

    """init buffer"""
    if args.if_off_policy:
        buffer = ReplayBuffer(
            gpu_id=args.gpu_id,
            num_seqs=args.num_envs,
            max_size=args.buffer_size,
            state_dim=args.state_dim,
            action_dim=1 if args.if_discrete else args.action_dim,
        )
        buffer_items = agent.explore_env(
            env, args.horizon_len * args.eval_times, if_random=True
        )
        buffer.update(buffer_items)  # warm up for ReplayBuffer
    else:
        buffer = []

    """init evaluator"""
    eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
    eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
    eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
    evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args)

    """train loop"""
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_off_policy = args.if_off_policy
    if_save_buffer = args.if_save_buffer
    del args

    import torch as th

    if_train = True
    while if_train:
        buffer_items = agent.explore_env(env, horizon_len)

        action = buffer_items[1].flatten()
        action_count = th.bincount(action).data.cpu().numpy() / action.shape[0]
        action_count = np.ceil(action_count * 998).astype(int)

        position = buffer_items[0][:, :, 0].long().flatten()
        position = position.float()  # TODO Only if on cpu
        position_count = torch.histc(
            position, bins=env.max_position * 2 + 1, min=-2, max=2
        )
        position_count = position_count.data.cpu().numpy() / position.shape[0]
        position_count = np.ceil(position_count * 998).astype(int)

        print(";;;", " " * 70, action_count, position_count)

        exp_r = buffer_items[2].mean().item()
        if if_off_policy:
            buffer.update(buffer_items)
        else:
            buffer[:] = buffer_items

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(
            actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_tuple=logging_tuple
        )
        if_train = (evaluator.total_step <= break_step) and (
            not os.path.exists(f"{cwd}/stop")
        )

    print(f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}")

    env.close() if hasattr(env, "close") else None
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)
    if if_save_buffer and hasattr(buffer, "save_or_load_history"):
        buffer.save_or_load_history(cwd, if_save=True)


def valid_agent(args: Config):
    cwd = f"{args.env_name}_D3QN_{args.gpu_id}"  # args.cwd
    thresh = 0.001

    eval_env_class = args.eval_env_class
    eval_env_args = args.eval_env_args

    agent_class = args.agent_class
    net_dims = args.net_dims

    sim: TradeSimulator = build_env(eval_env_class, eval_env_args, gpu_id=args.gpu_id)

    state_dim = eval_env_args["state_dim"]
    action_dim = eval_env_args["action_dim"]
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=args.gpu_id)

    agent.save_or_load_agent(cwd=cwd, if_save=False)
    agent_path = sorted(
        [
            file
            for file in os.listdir(cwd)
            if len(file) == len("actor_00154050_000.664.pth")
        ]
    )[-1]
    # agent_path = sorted([file for file in os.listdir(cwd)
    #                      if len(file) == len('actor_00191970.pth')])[-1]
    agent.act.load_state_dict(
        torch.load(f"{cwd}/{agent_path}", map_location=agent.device).state_dict()
    )

    actor = agent.act
    device = agent.device
    del agent

    # 定义时间范围
    state = sim.reset()

    position_ary = []
    trade_ary = []
    q_values_ary = []
    for i in range(sim.max_step):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device)
        tensor_q_values = actor(tensor_state)
        tensor_action = tensor_q_values.argmax(dim=1)

        mask_zero_position = sim.position.eq(0)
        mask_q_values = (
            tensor_q_values.max(dim=1)[0] - tensor_q_values.mean(dim=1)
        ).lt(torch.where(tensor_action.eq(2), thresh, thresh))
        mask = torch.logical_and(mask_zero_position, mask_q_values)
        tensor_action[mask] = 1

        action = tensor_action.detach().cpu().unsqueeze(1)
        state, reward, done, info_dict = sim.step(action=action)

        trade_ary.append(sim.action_int.data.cpu().numpy())
        position_ary.append(sim.position.data.cpu().numpy())
        q_values_ary.append(tensor_q_values.data.cpu().numpy())

    save_path = "erl_run_valid_position.npy"
    position_ary = np.stack(position_ary, axis=0)
    np.save(save_path, position_ary)
    print(f"| save valid_position in {save_path}")


def run():
    import sys

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # 从命令行参数里获得GPU_ID

    from erl_agent import AgentD3QN

    num_sims = 512
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7

    max_step = (4800 - num_ignore_step) // step_gap

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 8 + 2,  # factor_dim + (position, holding)
        "action_dim": 3,  # long, 0, short
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
    }
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (128, 128, 128)

    args.gamma = 0.995
    args.explore_rate = 0.005
    args.state_value_tau = 0.01
    args.soft_update_tau = 2e-6
    args.learning_rate = 2e-6
    args.batch_size = 512
    args.break_step = int(32e4)
    args.break_step = int(32)
    args.buffer_size = int(max_step * 32)
    args.repeat_times = 2
    args.horizon_len = int(max_step * 4)
    args.eval_per_step = int(max_step)
    args.num_workers = 1
    args.save_gap = 8

    args.eval_env_class = EvalTradeSimulator
    args.eval_env_args = env_args.copy()

    train_agent(args=args)
    valid_agent(args=args)


if __name__ == "__main__":
    run()
