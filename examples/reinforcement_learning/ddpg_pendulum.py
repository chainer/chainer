#!/usr/bin/env python
"""Example code of DDPG on OpenAI Gym environments.

For DDPG, see: https://arxiv.org/abs/1509.02971
"""
from __future__ import print_function
from __future__ import division
import argparse
import collections
import copy
import random

import gym
import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import training
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

    def __call__(self, obs, action):
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

    def __call__(self, x):
        """Compute actions for given observations."""
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return squash(self.l2(h),
                      self.xp.asarray(self.action_low),
                      self.xp.asarray(self.action_high))

class Updater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer_Q, optimizer_policy, env, reward_scale, tau, noise_scale):
        super(Updater, self).__init__(train_iter, optimizer_Q)
        self.optimizer_policy = optimizer_policy
        self.env = env
        self.reward_scale = reward_scale
        self.tau = tau
        self.noise_scale = noise_scale
        self.Q = optimizer_Q.target
        self.policy = optimizer_policy.target
        self.target_Q = copy.deepcopy(self.Q)
        self.target_policy = copy.deepcopy(self.policy)
        self.Rs = collections.deque(maxlen=100)  # History of returns
        self.episode = 0

    def update_core(self):
        obs = self.env.reset()
        done = False
        R = 0.0  # Return (sum of rewards obtained in an episode)
        timestep = 0

        train_iter = self.get_iterator('main')
        optimizer_Q = self.get_optimizer('main')
        optimizer_policy = self.optimizer_policy
        env = self.env
        Q = self.Q
        policy = self.policy
        target_Q = self.target_Q
        target_policy = self.target_policy

        while not done and timestep < env.spec.timestep_limit:
            # Select an action with additive noises for exploration
            action = (get_action(policy, obs) +
                      np.random.normal(scale=self.noise_scale))

            # Execute an action
            new_obs, reward, done, _ = env.step(
                np.clip(action, env.action_space.low, env.action_space.high))
            R += reward

            # Store a transition
            train_iter.D.append((obs, action, reward * self.reward_scale, done, new_obs))
            obs = new_obs

            # Sample a random minibatch of transitions and replay
            batch = train_iter.__next__()
            update(Q, target_Q, policy, target_policy,
                       optimizer_Q, optimizer_policy, batch)

            # Soft update of the target networks
            soft_copy_params(Q, target_Q, self.tau)
            soft_copy_params(policy, target_policy, self.tau)

            timestep += 1

        self.Rs.append(R)
        average_R = np.mean(self.Rs)
        print('episode: {} iteration: {} R:{} average_R:{}'.format(
              self.episode, train_iter.iteration, R, average_R))
        self.episode += 1


class GymIterator(chainer.dataset.Iterator):

    def __init__(self, batch_size, replay_start_size, timestep_limit):
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.timestep_limit = timestep_limit
        self.D = collections.deque(maxlen=10 ** 6)  # Replay buffer
        self.iteration = 0

    def __next__(self):
        self.iteration += 1
        if len(self.D) >= self.replay_start_size:
            batch_indices = random.sample(range(len(self.D)), self.batch_size)
            batch = [self.D[i] for i in batch_indices]
            return batch
        return self.D

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration / self.timestep_limit

def get_action(policy, obs):
    """Get an action by evaluating a given policy."""
    obs = policy.xp.asarray(obs[None], dtype=np.float32)
    with chainer.no_backprop_mode():
        action = policy(obs).data[0]
    return chainer.cuda.to_cpu(action)


def update(Q, target_Q, policy, target_policy, opt_Q, opt_policy,
           samples, gamma=0.99):
    """Update a Q-function and a policy."""
    xp = Q.xp
    obs = xp.asarray([sample[0] for sample in samples], dtype=np.float32)
    action = xp.asarray([sample[1] for sample in samples], dtype=np.float32)
    reward = xp.asarray([sample[2] for sample in samples], dtype=np.float32)
    done = xp.asarray([sample[3] for sample in samples], dtype=np.float32)
    obs_next = xp.asarray([sample[4] for sample in samples], dtype=np.float32)

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
        t.data[:] += tau * (s.data - t.data)


def main():

    parser = argparse.ArgumentParser(description='Chainer example: DDPG')
    parser.add_argument('--env', type=str, default='Pendulum-v0',
                        help='Name of the OpenAI Gym environment')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Number of transitions in each mini-batch')
    parser.add_argument('--episodes', '-e', type=int, default=1000,
                        help='Number of episodes to run')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
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
    args = parser.parse_args()

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
    train_iter = GymIterator(args.batch_size, args.replay_start_size, env.spec.timestep_limit)

    # Initialize models and optimizers
    Q = QFunction(obs_size, action_size, n_units=args.unit)
    policy = Policy(obs_size, action_size,
                    env.action_space.low, env.action_space.high,
                    n_units=args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        Q.to_gpu(args.gpu)
        policy.to_gpu(args.gpu)

    optimizer_Q = optimizers.Adam()
    optimizer_Q.setup(Q)
    optimizer_policy = optimizers.Adam(alpha=1e-4)
    optimizer_policy.setup(policy)

    # Set up a trainer
    updater = Updater(train_iter, optimizer_Q, optimizer_policy, env,
            args.reward_scale, args.tau, args.noise_scale)
    trainer = training.Trainer(updater, (args.episodes, 'epoch'))
    trainer.run()


if __name__ == '__main__':
    main()
