import sys
import torch as th
import numpy as np
import pandas as pd
from data_config import ConfigData


class TradeSimulator:
    def __init__(
        self,
        num_sims=64,
        slippage=5e-5,
        max_position=2,
        step_gap=1,
        delay_step=1,
        num_ignore_step=60,
        device=th.device("cpu"),
        gpu_id=-1,
    ):
        self.device = th.device(f"cuda:{gpu_id}") if gpu_id >= 0 else device
        self.num_sims = num_sims

        self.slippage = slippage
        self.delay_step = delay_step
        self.max_holding = 60 * 60 // step_gap
        self.max_position = max_position
        self.step_gap = step_gap
        self.sim_ids = th.arange(self.num_sims, device=self.device)

        """config"""
        args = ConfigData()

        """load data"""
        self.factor_ary = np.load(args.predict_ary_path)
        self.factor_ary = th.tensor(self.factor_ary, dtype=th.float32)  # CPU

        data_df = pd.read_csv(args.csv_path)  # CSV READ HERE

        self.price_ary = data_df[["bids_distance_3", "asks_distance_3", "midpoint"]].values
        self.price_ary[:, 0] = self.price_ary[:, 2] * (1 + self.price_ary[:, 0])
        self.price_ary[:, 1] = self.price_ary[:, 2] * (1 + self.price_ary[:, 1])

        """Align with the rear of the dataset instead"""
        # self.price_ary = self.price_ary[: self.factor_ary.shape[0], :]
        self.price_ary = self.price_ary[-self.factor_ary.shape[0] :, :]

        self.price_ary = th.tensor(self.price_ary, dtype=th.float32)  # CPU

        self.seq_len = 3600
        self.full_seq_len = self.price_ary.shape[0]
        assert self.price_ary.shape[0] == self.factor_ary.shape[0]

        # reset()
        self.step_i = 0
        self.step_is = th.zeros((num_sims,), dtype=th.long, device=device)
        self.action_int = th.zeros((num_sims,), dtype=th.long, device=device)
        self.rolling_asset = th.zeros((num_sims,), dtype=th.long, device=device)

        self.position = th.zeros((num_sims,), dtype=th.long, device=device)
        self.holding = th.zeros((num_sims,), dtype=th.long, device=device)
        self.empty_count = th.zeros((num_sims,), dtype=th.long, device=device)

        self.cash = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.asset = th.zeros((num_sims,), dtype=th.float32, device=device)

        # environment information
        self.env_name = "TradeSimulator-v0"
        self.state_dim = 8 + 2  # factor_dim + (position, holding)
        self.action_dim = 3  # short, nothing, long
        self.if_discrete = True
        self.max_step = (self.seq_len - num_ignore_step) // step_gap
        self.target_return = +np.inf

        """stop-loss"""
        self.best_price = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.stop_loss_thresh = 1e-3

    def _reset(self, slippage=None, _if_random=True):
        self.slippage = slippage if isinstance(slippage, float) else self.slippage

        num_sims = self.num_sims
        device = self.device

        # if if_random:
        i0s = np.random.randint(self.seq_len, self.full_seq_len - self.seq_len * 2, size=self.num_sims)
        self.step_i = 0
        self.step_is = th.tensor(i0s, dtype=th.long, device=self.device)
        self.cash = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.asset = th.zeros((num_sims,), dtype=th.float32, device=device)

        self.holding = th.zeros((num_sims,), dtype=th.long, device=device)
        self.position = th.zeros((num_sims,), dtype=th.long, device=device)
        self.empty_count = th.zeros((num_sims,), dtype=th.long, device=device)

        """stop-loss"""
        self.best_price = th.zeros((self.num_sims,), dtype=th.float32, device=self.device)

        step_is = self.step_is + self.step_i
        state = self.get_state(step_is_cpu=step_is.to(th.device("cpu")))
        return state

    def _step(self, action, _if_random=True):
        self.step_i += self.step_gap
        step_is = self.step_is + self.step_i
        step_is_cpu = step_is.to(th.device("cpu"))

        action = action.squeeze(1).to(self.device)
        action_int = action - 1  # map (0, 1, 2) to (-1, 0, +1), means (sell, nothing, buy)
        # action_int = (action - self.max_position) - self.position
        del action

        old_cash = self.cash
        old_asset = self.asset
        old_position = self.position

        # the data in price_ary is ['bid', 'ask', 'mid']
        # bid_price = self.price_ary[step_is_cpu, 0].to(self.device)
        # ask_price = self.price_ary[step_is_cpu, 1].to(self.device)
        mid_price = self.price_ary[step_is_cpu, 2].to(self.device)

        """get action_int"""
        truncated = self.step_i >= (self.max_step * self.step_gap)
        if truncated:
            action_int = -old_position
        else:
            new_position = (old_position + action_int).clip(
                -self.max_position, self.max_position
            )  # limit the position
            action_int = new_position - old_position  # get the limit action

            done_mask = (new_position * old_position).lt(0) & old_position.ne(0)
            if done_mask.sum() > 0:
                action_int[done_mask] = -old_position[done_mask]

        """holding"""
        self.holding = self.holding + 1
        mask_max_holding = self.holding.gt(self.max_holding)

        if mask_max_holding.sum() > 0:
            action_int[mask_max_holding] = -old_position[mask_max_holding]
        self.holding[old_position == 0] = 0

        # mask_min_holding = th.logical_and(self.holding.le(self.min_holding), old_position.ne(0))
        # if mask_min_holding.sum() > 0:
        #     action_int[mask_min_holding] = 0

        """stop-loss"""
        direction_mask1 = old_position.gt(0)
        if direction_mask1.sum() > 0:
            _best_price = th.max(
                th.stack([self.best_price[direction_mask1], mid_price[direction_mask1]]),
                dim=0,
            )[0]
            self.best_price[direction_mask1] = _best_price

        direction_mask2 = old_position.lt(0)
        if direction_mask2.sum() > 0:
            _best_price = th.min(
                th.stack([self.best_price[direction_mask2], mid_price[direction_mask2]]),
                dim=0,
            )[0]
            self.best_price[direction_mask2] = _best_price

        # stop_loss_thresh = mid_price * self.stop_loss_rate
        stop_loss_mask1 = th.logical_and(direction_mask1, (self.best_price - mid_price).gt(self.stop_loss_thresh))
        stop_loss_mask2 = th.logical_and(direction_mask2, (mid_price - self.best_price).gt(self.stop_loss_thresh))
        stop_loss_mask = th.logical_or(stop_loss_mask1, stop_loss_mask2)
        if stop_loss_mask.sum() > 0:
            action_int[stop_loss_mask] = -old_position[stop_loss_mask]

        """get new_position via action_int"""
        new_position = old_position + action_int

        entry_mask = old_position.eq(0)
        if entry_mask.sum() > 0:
            self.best_price[entry_mask] = mid_price[entry_mask]

        """executing"""
        direction = action_int.gt(0)  # True: buy, False: sell
        cost = action_int * mid_price  # action_int * th.where(direction, ask_price, bid_price)

        new_cash = old_cash - cost * th.where(direction, 1 + self.slippage, 1 - self.slippage)
        new_asset = new_cash + new_position * mid_price

        reward = new_asset - old_asset

        self.cash = new_cash  # update the cash
        self.asset = new_asset  # update the total asset
        self.position = new_position  # update the position
        self.action_int = action_int  # update the action_int

        state = self.get_state(step_is_cpu)
        info_dict = {}
        if truncated:
            terminal = th.ones_like(self.position, dtype=th.bool)
            state = self.reset()
        else:
            # terminal = old_position.ne(0) & new_position.eq(0)
            terminal = th.zeros_like(self.position, dtype=th.bool)

        return state, reward, terminal, info_dict

    def reset(self, slippage=None, date_strs=()):
        return self._reset(slippage=slippage, _if_random=True)

    def step(self, action):
        return self._step(action, _if_random=True)

    def get_state(self, step_is_cpu):
        factor_ary = self.factor_ary[step_is_cpu, :].to(self.device)

        return th.concat(
            (
                (self.position.float() / self.max_position)[:, None],
                (self.holding.float() / self.max_holding)[:, None],
                # (self.empty_count.float() / self.max_empty_count)[:, None],
                factor_ary,
            ),
            dim=1,
        )


class EvalTradeSimulator(TradeSimulator):

    def reset(self, slippage=None, date_strs=()):
        self.stop_loss_thresh = 1e-4
        return self._reset(slippage=slippage, _if_random=False)

    def step(self, action):
        return self._step(action, _if_random=False)


def check_simulator():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # 从命令行参数里获得GPU_ID
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    num_sims = 6
    slippage = 0
    step_gap = 2

    sim = TradeSimulator(num_sims=num_sims, step_gap=step_gap, slippage=slippage)
    action_dim = sim.action_dim
    delay_step = sim.delay_step

    reward_ary = th.zeros((num_sims, 4800), dtype=th.float32, device=device)

    state = sim.reset(slippage=slippage)
    for step_i in range(sim.max_step):
        action = th.randint(action_dim, size=(num_sims, 1), device=device)
        state, reward, done, info_dict = sim.step(action=action)

        reward_ary[:, step_i + delay_step] = reward

        print(sim.asset)  #  if step_i + 2 == sim.max_step else None

    print(reward_ary.sum(dim=1))
    print(state.shape, num_sims, sim.state_dim)
    assert state.shape == (num_sims, sim.state_dim)

    print("############")

    reward_ary = th.zeros((num_sims, sim.max_step + delay_step), dtype=th.float32, device=device)

    state = sim.reset(slippage=slippage)
    for step_i in range(sim.max_step):
        if step_i == 0:
            action = th.ones(size=(num_sims, 1), dtype=th.long, device=device) - 1
        else:
            action = th.ones(size=(num_sims, 1), dtype=th.long, device=device)

        state, reward, done, info_dict = sim.step(action=action)

        reward_ary[:, step_i + delay_step] = reward

        print(sim.asset) if step_i + 2 == sim.max_step else None

    print(reward_ary.sum(dim=1))
    print(state.shape, num_sims, sim.state_dim)
    assert state.shape == (num_sims, sim.state_dim)

    print()


if __name__ == "__main__":
    check_simulator()
