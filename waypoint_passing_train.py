import argparse
import os
import pickle
import shutil

from envs.scenario1_env import HoverEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from datetime import datetime
from utils.config import process_config


def main():
    envkey = {
        "s1_waypoint_passing": "s1_env_config",
    }
    trainkey = {
        "s1_waypoint_passing": "s1_train_config",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="s1_waypoint_passing")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=800)
    args = parser.parse_args()

    gs.init(logging_level="error")

    # 带时间戳的日志目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = f"logs/{args.exp_name}_{timestamp}"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = process_config(
        args, current_dir, envkey, trainkey
    )

    if args.vis:
        env_cfg["visualize_target"] = True

    env = HoverEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print("model saved to {}".format(os.path.join(log_dir, 'model_{}.pt'.format(runner.current_learning_iteration))))

if __name__ == "__main__":
    main()