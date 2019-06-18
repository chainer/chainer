#!/usr/bin/env python
"""Example code of DQN and DoubleDQN on OpenAI Gym environments.

For DQN, see: https://www.nature.com/articles/nature14236
For DoubleDQN, see: https://arxiv.org/abs/1509.06461
"""
from __future__ import division
import argparse
import collections
import copy
import random
import warnings

import gym
import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers


class QFunction(chainer.Chain):
    """Q-function represented by a MLP."""

    def __init__(self, obs_size, n_actions, n_units=100):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_units)
            self.l1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, n_actions)

    def forward(self, x):
        """Compute Q-values of actions for given observations."""
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return self.l2(h)


def get_greedy_action(Q, obs):
    """Get a greedy action wrt a given Q-function."""
    dtype = chainer.get_dtype()
    obs = Q.xp.asarray(obs[None], dtype=dtype)
    with chainer.no_backprop_mode():
        q = Q(obs).array[0]
    return int(q.argmax())


def mean_clipped_loss(y, t):
    return F.mean(F.huber_loss(y, t, delta=1.0, reduce='no'))


def update(Q, target_Q, opt, samples, gamma=0.99, target_type='double_dqn'):
    """Update a Q-function with given samples and a target Q-function."""
    dtype = chainer.get_dtype()
    xp = Q.xp
    obs = xp.asarray([sample[0] for sample in samples], dtype=dtype)
    action = xp.asarray([sample[1] for sample in samples], dtype=np.int32)
    reward = xp.asarray([sample[2] for sample in samples], dtype=dtype)
    done = xp.asarray([sample[3] for sample in samples], dtype=dtype)
    obs_next = xp.asarray([sample[4] for sample in samples], dtype=dtype)
    # Predicted values: Q(s,a)
    y = F.select_item(Q(obs), action)
    # Target values: r + gamma * max_b Q(s',b)
    with chainer.no_backprop_mode():
        if target_type == 'dqn':
            next_q = F.max(target_Q(obs_next), axis=1)
        elif target_type == 'double_dqn':
            next_q = F.select_item(target_Q(obs_next),
                                   F.argmax(Q(obs_next), axis=1))
        else:
            raise ValueError('Unsupported target_type: {}'.format(target_type))
        target = reward + gamma * (1 - done) * next_q
    loss = mean_clipped_loss(y, target)
    Q.cleargrads()
    loss.backward()
    opt.update()


def main():

    parser = argparse.ArgumentParser(description='Chainer example: DQN')
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Name of the OpenAI Gym environment')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Number of transitions in each mini-batch')
    parser.add_argument('--episodes', '-e', type=int, default=1000,
                        help='Number of episodes to run')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='dqn_result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    parser.add_argument('--target-type', type=str, default='dqn',
                        help='Target type', choices=['dqn', 'double_dqn'])
    parser.add_argument('--reward-scale', type=float, default=1e-2,
                        help='Reward scale factor')
    parser.add_argument('--replay-start-size', type=int, default=500,
                        help=('Number of iterations after which replay is '
                              'started'))
    parser.add_argument('--iterations-to-decay-epsilon', type=int,
                        default=5000,
                        help='Number of steps used to linearly decay epsilon')
    parser.add_argument('--min-epsilon', type=float, default=0.01,
                        help='Minimum value of epsilon')
    parser.add_argument('--target-update-freq', type=int, default=100,
                        help='Frequency of target network update')
    parser.add_argument('--record', action='store_true', default=True,
                        help='Record performance')
    parser.add_argument('--no-record', action='store_false', dest='record')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if chainer.get_dtype() == np.float16:
        warnings.warn(
            'This example may cause NaN in FP16 mode.', RuntimeWarning)

    device = chainer.get_device(args.device)
    device.use()

    # Initialize an environment
    env = gym.make(args.env)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    obs_size = env.observation_space.low.size
    n_actions = env.action_space.n
    if args.record:
        env = gym.wrappers.Monitor(env, args.out, force=True)
    reward_threshold = env.spec.reward_threshold
    if reward_threshold is not None:
        print('{} defines "solving" as getting average reward of {} over 100 '
              'consecutive trials.'.format(args.env, reward_threshold))
    else:
        print('{} is an unsolved environment, which means it does not have a '
              'specified reward threshold at which it\'s considered '
              'solved.'.format(args.env))

    # Initialize variables
    D = collections.deque(maxlen=10 ** 6)  # Replay buffer
    Rs = collections.deque(maxlen=100)  # History of returns
    iteration = 0

    # Initialize a model and its optimizer
    Q = QFunction(obs_size, n_actions, n_units=args.unit)
    Q.to_device(device)
    target_Q = copy.deepcopy(Q)
    opt = optimizers.Adam(eps=1e-2)
    opt.setup(Q)

    for episode in range(args.episodes):

        obs = env.reset()
        done = False
        R = 0.0  # Return (sum of rewards obtained in an episode)
        timestep = 0

        while not done and timestep < env.spec.timestep_limit:

            # Epsilon is linearly decayed
            epsilon = 1.0 if len(D) < args.replay_start_size else \
                max(args.min_epsilon,
                    np.interp(
                        iteration,
                        [0, args.iterations_to_decay_epsilon],
                        [1.0, args.min_epsilon]))

            # Select an action epsilon-greedily
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = get_greedy_action(Q, obs)

            # Execute an action
            new_obs, reward, done, _ = env.step(action)
            R += reward

            # Store a transition
            D.append((obs, action, reward * args.reward_scale, done, new_obs))
            obs = new_obs

            # Sample a random minibatch of transitions and replay
            if len(D) >= args.replay_start_size:
                sample_indices = random.sample(range(len(D)), args.batch_size)
                samples = [D[i] for i in sample_indices]
                update(Q, target_Q, opt, samples, target_type=args.target_type)

            # Update the target network
            if iteration % args.target_update_freq == 0:
                target_Q = copy.deepcopy(Q)

            iteration += 1
            timestep += 1

        Rs.append(R)
        average_R = np.mean(Rs)
        print('episode: {} iteration: {} R: {} average_R: {}'.format(
              episode, iteration, R, average_R))

        if reward_threshold is not None and average_R >= reward_threshold:
            print('Solved {} by getting average reward of '
                  '{} >= {} over 100 consecutive episodes.'.format(
                      args.env, average_R, reward_threshold))
            break


if __name__ == '__main__':
    main()
