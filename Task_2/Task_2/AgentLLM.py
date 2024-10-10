"""Llama3 8b based agent. 
The agent reads in natural language text in the form of headline news and 
predicts an average trajectory for the sp500 for an upcoming period."""

import torch
import torch.nn as nn
from typing import Tuple, List
from torch import Tensor

from elegantrl.train.config import Config
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorPPO, CriticPPO, ActorBase, CriticBase

from elegantrl.agents import AgentPPO, AgentA2C, AgentDDPG, AgentSAC

from transformers import AutoModelForCausalLM, AutoTokenizer


class ActorLLM(ActorBase):
    def __init__(self, model_name: str, state_dim: int, action_dim: int, args: Config):
        # initialize a base actor -- maybe remove later
        super().__init__(state_dim=state_dim, action_dim=action_dim)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, return_dict=True, output_hidden_states=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Code to freeze params
        # for param in self.model.parameters():
        #     param.requires_grad = False

        hidden_size = self.model.config.hidden_size

        # Actor Head: Maps LLM outputs to action means
        self.actor_mean = nn.Linear(hidden_size, action_dim)

        # Log standard deviation: One log std per action dimension
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Initialize the actor head with orthogonal weights
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.1)
        nn.init.constant_(self.actor_mean.bias, 0)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Compute the action means using the LLM.

        Args:
            input_ids (Tensor): Tokenized NL inputs.
            attention_mask (Tensor): Attention masks.

        Returns:
            Tensor: Action means, shape (batch_size, action_dim)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # No [CLS] token so use first one
        # Shape: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

        action_mean = self.actor_mean(pooled_output)  # Shape: (batch_size, action_dim)
        return torch.tanh(action_mean)  # Bound actions between -1 and 1

    def get_action(self, input_ids: Tensor, attention_mask: Tensor) -> (Tensor, Tensor):
        """
        Sample an action and compute its log probability for a given state.

        Args:
            input_ids (Tensor): Tokenized NL inputs.
            attention_mask (Tensor): Attention masks.

        Returns:
            Tuple[Tensor, Tensor]: Sampled actions and their log probabilities.
        """
        # Shape: (batch_size, action_dim)
        action_mean = self.forward(input_ids, attention_mask)
        # Shape: (1, action_dim)
        action_std = torch.exp(self.action_log_std)

        # Expand action_std to match batch size
        # Shape: (batch_size, action_dim)
        action_std = action_std.expand_as(action_mean)

        # Define the action distribution
        dist = torch.distributions.Normal(action_mean, action_std)

        # Sample actions
        action = dist.rsample()  # rsample for reparameterization trick

        # Compute log probabilities
        log_prob = dist.log_prob(action).sum(dim=1)  # Shape: (batch_size,)

        return action, log_prob

    def get_logprob_entropy(
        self, input_ids: Tensor, attention_mask: Tensor, action: Tensor
    ) -> (Tensor, Tensor):
        """
        Compute log probabilities and entropy for given actions.

        Args:
            input_ids (Tensor): Tokenized NL inputs.
            attention_mask (Tensor): Attention masks.
            action (Tensor): Actions taken.

        Returns:
            Tuple[Tensor, Tensor]: Log probabilities and entropy.
        """
        action_mean = self.forward(
            input_ids, attention_mask
        )  # Shape: (batch_size, action_dim)
        action_std = torch.exp(self.action_log_std)  # Shape: (1, action_dim)

        # Expand action_std to match batch size
        # Shape: (batch_size, action_dim)
        action_std = action_std.expand_as(action_mean)

        # Define the action distribution
        dist = torch.distributions.Normal(action_mean, action_std)

        # Compute log probabilities
        log_prob = dist.log_prob(action).sum(dim=1)  # Shape: (batch_size,)

        # Compute entropy
        entropy = dist.entropy().sum(dim=1)  # Shape: (batch_size,)

        return log_prob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        # Already bounded by tanh in forward; modify if necessary
        return action


# Use a CriticPPO for the critic


class AgentLLM(AgentBase):
    def __init__(
        self,
        model_name: str,
        net_dims: List[int],
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        args: Config = Config(),
    ):
        """from base agent"""
        self.gamma = args.gamma  # discount factor of future rewards
        self.num_envs = (
            args.num_envs
        )  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = (
            args.batch_size
        )  # num of transitions sampled from replay buffer.
        self.repeat_times = (
            args.repeat_times
        )  # repeatedly update network using ReplayBuffer
        self.reward_scale = (
            args.reward_scale
        )  # an approximate target reward usually be closed to 256
        self.learning_rate = (
            args.learning_rate
        )  # the learning rate for network updating
        self.if_off_policy = (
            args.if_off_policy
        )  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = (
            args.clip_grad_norm
        )  # clip the gradient after normalization
        self.soft_update_tau = (
            args.soft_update_tau
        )  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = (
            args.state_value_tau
        )  # the tau of normalize for value and state

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )

        """custom llm agent"""
        self.model_name = model_name
        self.act = ActorLLM(model_name, state_dim, action_dim, args).to(self.device)
        self.cri = CriticPPO(net_dims, state_dim, action_dim).to(self.device)

        self.tokenizer = self.act.tokenizer  # Access tokenizer from actor

        """optimizer"""
        self.act_optimizer = torch.optim.AdamW(
            self.act.parameters(), self.learning_rate
        )
        self.cri_optimizer = torch.optim.AdamW(
            self.cri.parameters(), self.learning_rate
        )
        from types import MethodType  # built-in package of Python3

        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        """attribute"""
        if self.num_envs == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        self.if_use_per = getattr(
            args, "if_use_per", None
        )  # use PER (Prioritized Experience Replay)
        if self.if_use_per:
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

        """save and load"""
        self.save_attr_names = {
            "act",
            "act_target",
            "act_optimizer",
            "cri",
            "cri_target",
            "cri_optimizer",
        }

    def explore_one_env(
        self, env, horizon_len: int, if_random: bool = False
    ) -> Tuple[Tensor, ...]:
        """
        Collect trajectories using an LLM-based actor for a single environment.

        Args:
            env: RL training environment.
            horizon_len (int): Number of steps to collect.
            if_random (bool): Whether to use random actions.

        Notes:
            The state that this operates on is a tensor with NL inputs

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: states, actions, rewards, undones
        """

        # Init tensors TODO convert to torch or np depnding on memory
        states = torch.zeros(
            (horizon_len, self.num_envs, self.state_dim), dtype=torch.float32
        ).to(self.device)
        actions = torch.zeros(
            (horizon_len, self.num_envs, self.action_dim), dtype=torch.float32
        ).to(self.device)
        logprobs = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(
            self.device
        )
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(
            self.device
        )
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(
            self.device
        )

        # should init to None
        state = self.last_state

        # This çalls the forward method in the LLM actor class
        get_action = self.act.get_action
        # shouldn't be doing anything
        convert = self.act.convert_action_for_env

        # horizon len exploration loop
        for t in range(horizon_len):
            nl_input, market_data = state["nl_input"], state["market_data"]

            inputs = self.tokenizer(
                [nl_input],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # calls the AgentLLM getaction function
            if if_random:
                action = (
                    torch.rand(self.num_envs, self.action_dim).to(self.device) * 2 - 1.0
                )
                logprob = torch.zeros(self.num_envs, dtype=torch.float32).to(
                    self.device
                )
            else:
                action, logprob = get_action(input_ids, attention_mask)

            ary_action = convert(action[0]).detach().cpu().numpy()
            env_state, reward, done, _ = env.step(ary_action)  # next_state
            env_state = env.reset() if done else env_state

            # the env should return next nl_input and market data
            state = {
                "nl_input": env_state["nl_input"],
                "market_data": env_state["market_data"],
            }

            market_tensor = torch.as_tensor(market_data, dtype=torch.float32).to(
                self.device
            )
            states[t] = market_tensor.unsqueeze(0)  # Shape: (num_envs=1, state_dim)
            actions[t] = action
            logprobs[t] = logprob
            rewards[t] = reward
            dones[t] = done

        self.last_state = state  # state.shape == (1, state_dim) for a single env.

        # rewards *= self.reward_scale # maybe uncomment this at a later point
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, logprobs, rewards, undones

    def update_net(self, buffer) -> Tuple[float, ...]:

        states, actions, logprobs, rewards, undones = buffer
        buffer_size = states.shape[0]
        buffer_num = states.shape[1]

        """batch compute value estimations"""
        bs = 2**10  # Batch size for value computation
        values = torch.empty_like(rewards)  # values.shape == (buffer_size, buffer_num)
        for i in range(0, buffer_size, bs):
            end = min(i + bs, buffer_size)
            # Extract market_data for the critic
            market_data_batch = states[i:end, :, "market_data"]  # TODO states is a dict
            # Flatten batch for processing
            market_data_flat = market_data_batch.reshape(-1, self.state_dim)
            # Get value estimates
            value_flat = self.cri(
                market_data_flat, actions[i:end].reshape(-1, self.action_dim)
            )
            # Reshape back to (batch_size, buffer_num)
            value = value_flat.view(end - i, buffer_num)
            values[i:end] = value

        """compute advantages - how much better the actor did than the critic"""
        # shape == (buffer_size, buffer_num)
        advantages = self.get_advantages(rewards, undones, values)
        reward_sums = advantages + values  # shape == (buffer_size, buffer_num)
        del rewards, undones, values

        advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-4)

        self.update_avg_std_for_normalization(
            states=states.reshape((-1, self.state_dim)),
            returns=reward_sums.reshape((-1,)),
        )

        # assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, buffer_num)

        """PPO loop --> update network"""
        obj_critics = 0.0
        obj_actors = 0.0
        sample_len = buffer_size - 1

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            """For mini batch"""
            ids = torch.randint(
                sample_len * buffer_num, size=(self.batch_size,), requires_grad=False
            )
            # ids % sample_len
            ids0 = torch.fmod(ids, sample_len)
            # ids // sample_lens
            ids1 = torch.div(ids, sample_len, rounding_mode="floor")

            """Samples states, action, reward"""
            state = states[ids0, ids1]  # should be a dict: nl_input and market_data
            action = actions[ids0, ids1]
            logprob = logprobs[ids0, ids1]
            advantage = advantages[ids0, ids1]
            reward_sum = reward_sums[ids0, ids1]

            """critic network update --> critic estimates reward Q values of state"""
            # extract market data for the critic network - the env returns nl + market data
            market_data = state["market_data"]
            value = self.cri(market_data, reward_sum)
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            """actor network update --> here we do LLM stuff"""
            nl_input = state["nl_input"]
            if isinstance(nl_input, list):
                nl_input = nl_input
            else:
                nl_input = nl_input.tolist()
            tokenized_inputs = self.tokenizer(
                nl_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            ).to(self.device)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                input_ids=tokenized_inputs["input_ids"],
                attention_mask=tokenized_inputs["attention_mask"],
                actions=actions,
            )

            # Calculate PPO ratio
            ratio = (new_logprob - logprob.detach()).exp()  # Shape: (batch_size,)

            # Compute surrogate objectives
            surrogate1 = advantage * ratio  # Shape: (batch_size,)
            surrogate2 = advantage * torch.clamp(
                ratio, 1 - self.ratio_clip, 1 + self.ratio_clip
            )  # Shape: (batch_size,)

            # PPO surrogate loss
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            # Entropy bonus
            entropy_bonus = obj_entropy.mean()

            # Total actor loss
            obj_actor = (
                -obj_surrogate - self.lambda_entropy * entropy_bonus
            )  # Negative for gradient ascent
            # Update actor
            self.optimizer_update(self.act_optimizer, obj_actor)
            obj_actors += obj_actor.item()

        # Compute average losses
        avg_critic_loss = obj_critics / update_times
        avg_actor_loss = obj_actors / update_times

        # Log action standard deviation if applicable
        a_std_log = (
            self.act.action_log_std.mean().item()
            if hasattr(self.act, "action_log_std")
            else 0.0
        )

        return avg_critic_loss, avg_actor_loss, a_std_log


def get_optim_param(optimizer: torch.optim) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend(
            [t for t in params_dict.values() if isinstance(t, torch.Tensor)]
        )
    return params_list


def testActorLLM():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model parameters
    model_name = "gpt2"
    action_dim = 1  # Example: Predicting the next price

    # Initialize the ActorLLM
    print("Initializing ActorLLM...")
    actor = ActorLLM(model_name=model_name, action_dim=action_dim).to(device)
    actor.eval()  # Set to evaluation mode for testing

    # Sample Natural Language inputs
    nl_inputs = [
        "Breaking news: The stock market is experiencing unprecedented growth.",
        "Economic indicators suggest a potential recession in the coming months.",
    ]

    print("\nSample NL Inputs:")
    for i, text in enumerate(nl_inputs):
        print(f"{i+1}: {text}")

    # Tokenize the inputs
    print("\nTokenizing inputs...")
    tokenized_inputs = actor.tokenizer(
        nl_inputs,
        return_tensors="pt",  # Return PyTorch tensors
        padding=True,  # Pad sequences to the same length
        truncation=True,  # Truncate sequences longer than max_length
        max_length=512,  # Set a maximum sequence length
    )
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention Mask shape: {attention_mask.shape}")

    # Forward pass to get action means
    print("\nPerforming forward pass to obtain action means...")
    with torch.no_grad():
        action_means = actor.forward(input_ids, attention_mask)
    print(f"Action Means:\n{action_means}")

    # Get sampled actions and log probabilities
    print("\nSampling actions and computing log probabilities...")
    with torch.no_grad():
        actions, log_probs = actor.get_action(input_ids, attention_mask)
    print(f"Sampled Actions:\n{actions}")
    print(f"Log Probabilities:\n{log_probs}")

    # Compute log probabilities and entropy for the sampled actions
    print("\nComputing log probabilities and entropy for the sampled actions...")
    with torch.no_grad():
        computed_log_probs, entropy = actor.get_logprob_entropy(
            input_ids, attention_mask, actions
        )
    print(f"Computed Log Probabilities:\n{computed_log_probs}")
    print(f"Entropy:\n{entropy}")

    # Verify consistency
    print("\nVerifying consistency between sampled and computed log probabilities...")
    difference = (log_probs - computed_log_probs).abs().max().item()
    print(
        f"Maximum difference between sampled and computed log probabilities: {difference}"
    )

    if difference < 1e-6:
        print("Log probabilities are consistent.")
    else:
        print("Log probabilities are inconsistent. Check implementation.")

    # Optional: Convert actions for environment
    print("\nConverting actions for environment...")
    env_actions = ActorLLM.convert_action_for_env(actions)
    print(f"Environment-Compatible Actions:\n{env_actions}")


def testAgentLLM():
    model_name = "bert-base-uncased"  # Using a smaller model for testing
    net_dims = [256, 256]  # can be anything really we throw this out in actor llm
    state_dim = 5
    action_dim = 1
    gpu_id = 0 if torch.cuda.is_available() else -1
    args = Config()

    agent = AgentLLM(
        model_name=model_name,
        net_dims=net_dims,
        state_dim=state_dim,
        action_dim=action_dim,
        gpu_id=gpu_id,
        args=args,
    )


def main():
    # testActorLLM()
    testAgentLLM()


if __name__ == "__main__":
    main()
