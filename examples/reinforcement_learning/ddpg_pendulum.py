#!/usr/bin/env python
"""Example code of DDPG on OpenAI Gym environments.

For DDPG, see: https://arxiv.org/abs/1509.02971
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

    def __init__(self, obs_size, action_size, n_units=100):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size + action_size, n_units)
            self.l1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, 1,
                               initialW=chainer.initializers.HeNormal(1e-3))

    def forward(self, obs, action):
        """Compute Q-values for given state-action pairs."""
        x = F.concat((obs, action), axis=1)
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return self.l2(h)


def squash(x, low, high):
    """Squash values to fit [low, high] via tanh."""
    center = (high + low) / 2
    scale = (high - low) / 2
    return F.tanh(x) * scale + center


class Policy(chainer.Chain):
    """Policy represented by a MLP."""

    def __init__(self, obs_size, action_size, action_low, action_high,
                 n_units=100):
        super(Policy, self).__init__()
        self.action_high = action_high
        self.action_low = action_low
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_units)
            self.l1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, action_size,
                               initialW=chainer.initializers.HeNormal(1e-3))

    def forward(self, x):
        """Compute actions for given observations."""
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return squash(self.l2(h),
                      self.xp.asarray(self.action_low),
                      self.xp.asarray(self.action_high))


def get_action(policy, obs):
    """Get an action by evaluating a given policy."""
    dtype = chainer.get_dtype()
    obs = policy.xp.asarray(obs[None], dtype=dtype)
    with chainer.no_backprop_mode():
        action = policy(obs).array[0]
    return chainer.backends.cuda.to_cpu(action)


def update(Q, target_Q, policy, target_policy, opt_Q, opt_policy,
           samples, gamma=0.99):
    """Update a Q-function and a policy."""
    dtype = chainer.get_dtype()
    xp = Q.xp
    obs = xp.asarray([sample[0] for sample in samples], dtype=dtype)
    action = xp.asarray([sample[1] for sample in samples], dtype=dtype)
    reward = xp.asarray([sample[2] for sample in samples], dtype=dtype)
    done = xp.asarray([sample[3] for sample in samples], dtype=dtype)
    obs_next = xp.asarray([sample[4] for sample in samples], dtype=dtype)

    def update_Q():
        # Predicted values: Q(s,a)
        y = F.squeeze(Q(obs, action), axis=1)
        # Target values: r + gamma * Q(s,policy(s))
        with chainer.no_backprop_mode():
            next_q = F.squeeze(target_Q(obs_next, target_policy(obs_next)),
                               axis=1)
            target = reward + gamma * (1 - done) * next_q
        loss = F.mean_squared_error(y, target)
        Q.cleargrads()
        loss.backward()
        opt_Q.update()

    def update_policy():
        # Maximize Q(s,policy(s))
        q = Q(obs, policy(obs))
        q = q[:]  # Avoid https://github.com/chainer/chainer/issues/2744
        loss = - F.mean(q)
        policy.cleargrads()
        loss.backward()
        opt_policy.update()

    update_Q()
    update_policy()


def soft_copy_params(source, target, tau):
    """Make the parameters of a link close to the ones of another link.

    Making tau close to 0 slows the pace of updates, and close to 1 might lead
    to faster, but more volatile updates.
    """
    # Sort params by name
    source_params = [param for _, param in sorted(source.namedparams())]
    target_params = [param for _, param in sorted(target.namedparams())]
    for s, t in zip(source_params, target_params):
        t.array[:] += tau * (s.array - t.array)


def main():

    parser = argparse.ArgumentParser(description='Chainer example: DDPG')
    parser.add_argument('--env', type=str, default='Pendulum-v0',
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
    parser.add_argument('--out', '-o', default='ddpg_result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    parser.add_argument('--reward-scale', type=float, default=1e-3,
                        help='Reward scale factor')
    parser.add_argument('--replay-start-size', type=int, default=500,
                        help=('Number of iterations after which replay is '
                              'started'))
    parser.add_argument('--tau', type=float, default=1e-2,
                        help='Softness of soft target update (0, 1]')
    parser.add_argument('--noise-scale', type=float, default=0.4,
                        help='Scale of additive Gaussian noises')
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
    assert isinstance(env.action_space, gym.spaces.Box)
    obs_size = env.observation_space.low.size
    action_size = env.action_space.low.size
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

    # Initialize models and optimizers
    Q = QFunction(obs_size, action_size, n_units=args.unit)
    policy = Policy(obs_size, action_size,
                    env.action_space.low, env.action_space.high,
                    n_units=args.unit)
    Q.to_device(device)
    policy.to_device(device)
    target_Q = copy.deepcopy(Q)
    target_policy = copy.deepcopy(policy)
    opt_Q = optimizers.Adam(eps=1e-5)  # Use larger eps in case of FP16 mode
    opt_Q.setup(Q)
    opt_policy = optimizers.Adam(alpha=1e-4)
    opt_policy.setup(policy)

    for episode in range(args.episodes):

        obs = env.reset()
        done = False
        R = 0.0  # Return (sum of rewards obtained in an episode)
        timestep = 0

        while not done and timestep < env.spec.timestep_limit:

            # Select an action with additive noises for exploration
            action = (get_action(policy, obs) +
                      np.random.normal(scale=args.noise_scale))

            # Execute an action
            new_obs, reward, done, _ = env.step(
                np.clip(action, env.action_space.low, env.action_space.high))
            R += reward

            # Store a transition
            D.append((obs, action, reward * args.reward_scale, done, new_obs))
            obs = new_obs

            # Sample a random minibatch of transitions and replay
            if len(D) >= args.replay_start_size:
                sample_indices = random.sample(range(len(D)), args.batch_size)
                samples = [D[i] for i in sample_indices]
                update(Q, target_Q, policy, target_policy,
                       opt_Q, opt_policy, samples)

            # Soft update of the target networks
            soft_copy_params(Q, target_Q, args.tau)
            soft_copy_params(policy, target_policy, args.tau)

            iteration += 1
            timestep += 1

        Rs.append(R)
        average_R = np.mean(Rs)
        print('episode: {} iteration: {} R:{} average_R:{}'.format(
              episode, iteration, R, average_R))

        if reward_threshold is not None and average_R >= reward_threshold:
            print('Solved {} by getting average reward of '
                  '{} >= {} over 100 consecutive episodes.'.format(
                      args.env, average_R, reward_threshold))
            break


if __name__ == '__main__':
    main()
