"""PPO evaluation script for racecar_gym (4 agents, shared policy).

Usage:
    python eval_ppo.py --checkpoint log/PPO_4agents/final_model

Outputs (saved to --out_dir):
    eval_ep<N>.mp4   : video with per-agent overlay (step/lap/progress)
"""

import argparse
import os

import cv2
import numpy as np
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecTransposeImage

from ppo.multi_agent_vec_env import MultiAgentSharedPolicyVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to saved model (without .zip extension)')
    parser.add_argument('--scenario', type=str,
                        default='scenarios/austria_rgb_4agents.yml')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--n_frames', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=84)
    parser.add_argument('--video_fps', type=int, default=30)
    parser.add_argument('--render', type=str, default='human', \
                        choices=['human', 'rgb_array_follow', 'rgb_array_birds_eye'])
    parser.add_argument('--render_size', type=int, default=480)
    parser.add_argument('--out_dir', type=str, default='eval_output')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def make_eval_env(scenario: str, img_size: int, render: str, n_frames: int):
    vec_env = MultiAgentSharedPolicyVecEnv(
        scenario_path=scenario,
        img_size=img_size,
        render_mode=render,
    )
    vec_env = VecMonitor(vec_env)
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=n_frames)
    return vec_env


def draw_overlay(frame: np.ndarray, agent_ids, infos, step: int) -> np.ndarray:
    """Draw per-agent lap/progress text onto a BGR frame."""
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'step {step}', (10, 22), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for i, aid in enumerate(agent_ids):
        info = infos[i] if infos else {}
        lap = info.get('lap', 0)
        prog = info.get('progress', 0.0)
        cv2.putText(img, f'{aid}  lap:{lap}  prog:{prog:.3f}',
                    (10, 22 + (i + 1) * 22), font, 0.55, (0, 255, 180), 1, cv2.LINE_AA)
    return img


def make_agent_table(agent_ids, infos, rewards, total_rewards, step):
    table = Table(title=f"Step {step}", show_header=True, header_style="bold cyan")
    table.add_column("Agent", style="bold", width=8)
    table.add_column("Lap",   justify="center", width=5)
    table.add_column("Progress", justify="center", width=24)
    table.add_column("Reward", justify="right", width=9)
    table.add_column("Total Reward", justify="right", width=13)
    for i, aid in enumerate(agent_ids):
        info = infos[i] if infos is not None else {}
        lap = info.get('lap', 0)
        progress = info.get('progress', 0.0)
        bar = '█' * int(progress * 20) + '░' * (20 - int(progress * 20))
        table.add_row(
            aid, str(lap),
            f"[green]{bar}[/green] {progress:.3f}",
            f"{rewards[i]:.3f}",
            f"{total_rewards[i]:.2f}",
        )
    return table


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading model: {args.checkpoint}')
    env = make_eval_env(args.scenario, args.img_size, args.render, args.n_frames)
    model = PPO.load(args.checkpoint, env=env, device=args.device)

    base_env = env.venv.venv.venv
    agent_ids = base_env._agent_ids
    sz = args.render_size

    for ep in range(args.episodes):
        obs = env.reset()
        total_rewards = np.zeros(base_env._n_agents)
        infos = [{}] * base_env._n_agents
        rewards = np.zeros(base_env._n_agents)

        mp4_path = os.path.join(args.out_dir, f'eval_ep{ep + 1}.mp4')
        writer = cv2.VideoWriter(
            mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), args.video_fps, (sz, sz),
        )

        with Live(refresh_per_second=10) as live:
            for step in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                total_rewards += rewards

                raw = base_env.render()
                if raw is not None:
                    bgr = cv2.cvtColor(
                        cv2.resize(raw[:, :, :3].astype(np.uint8), (sz, sz)),
                        cv2.COLOR_RGB2BGR,
                    )
                    cv2.imshow('Evaluation', bgr)
                    cv2.waitKey(1)
                    writer.write(draw_overlay(bgr, agent_ids, infos, step))

                live.update(Panel(
                    make_agent_table(agent_ids, infos, rewards, total_rewards, step),
                    title=f"[bold]Episode {ep + 1}/{args.episodes}[/bold]",
                ))

                if dones.all():
                    break

        writer.release()
        print(f'\nEpisode {ep + 1}: mean_reward={total_rewards.mean():.2f}')
        print(f'  MP4 → {mp4_path}')

    env.close()


if __name__ == '__main__':
    main()

