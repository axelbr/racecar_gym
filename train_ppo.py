"""PPO training script for racecar_gym with 4 agents sharing a single policy.

Usage:
    python train_ppo.py
    python train_ppo.py --scenario scenarios/austria_rgb_4agents.yml --timesteps 5000000

The 4 agents share the same CnnPolicy and interact in the same physics simulation.
Observations: rgb_camera → grayscale → 84x84, stacked 4 frames.
Actions: [motor, steering] ∈ [-1, 1]².
"""

import argparse
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecTransposeImage

from ppo.multi_agent_vec_env import MultiAgentSharedPolicyVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str,
                        default='scenarios/austria_rgb_4agents.yml',
                        help='Path to scenario YAML')
    parser.add_argument('--timesteps', type=float, default=5e6,
                        help='Total training timesteps')
    parser.add_argument('--n_steps', type=int, default=1024,
                        help='Steps per rollout per agent')
    parser.add_argument('--n_frames', type=int, default=4,
                        help='Number of frames to stack')
    parser.add_argument('--img_size', type=int, default=84,
                        help='Observation image size (square)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='./log/PPO_4agents/')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='TensorBoard log dir (requires tensorboard package). Omit to disable.')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def make_env(scenario: str, img_size: int, n_frames: int):
    vec_env = MultiAgentSharedPolicyVecEnv(
        scenario_path=scenario,
        img_size=img_size,
        render_mode='rgb_array_birds_eye',  # headless rendering for training
    )
    vec_env = VecMonitor(vec_env)
    vec_env = VecTransposeImage(vec_env)    # (H,W,C) → (C,H,W) for CnnPolicy
    vec_env = VecFrameStack(vec_env, n_stack=n_frames)
    return vec_env


def main():
    args = parse_args()
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    print(f'Scenario: {args.scenario}')
    print(f'Total timesteps: {int(args.timesteps):,}')
    print(f'Checkpoint dir: {checkpoint_dir}')

    env = make_env(args.scenario, args.img_size, args.n_frames)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.n_steps * 10,    # save every 10 rollouts
        save_path=checkpoint_dir,
        name_prefix='ppo_racecar',
        save_replay_buffer=False,
    )

    model = PPO(
        policy='CnnPolicy',
        env=env,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        clip_range=0.2,
        use_sde=True,           # state-dependent exploration
        ent_coef=0.01,
    )

    print('Starting training...')
    model.learn(
        total_timesteps=int(args.timesteps),
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_path = os.path.join(checkpoint_dir, 'final_model')
    model.save(final_path)
    print(f'Training complete. Model saved to {final_path}.zip')
    env.close()


if __name__ == '__main__':
    main()
