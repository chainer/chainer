#!/usr/bin/env python
"""Example code of DQN and DoubleDQN on OpenAI Gym environments.

For DQN, see: http://www.nature.com/articles/nature14236
For DoubleDQN, see: https://arxiv.org/abs/1509.06461
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
from chainer import optimizers


class QFunction(chainer.Chain):
    """Q-function represented by a MLP."""

    def __init__(self, obs_size, n_actions, n_units=100):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_units)
            self.l1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, n_actions)

    def __call__(self, x):
        """Compute Q-values of actions for given observations."""
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return self.l2(h)


def get_greedy_action(Q, obs):
    xp = Q.xp
    obs = xp.expand_dims(xp.asarray(obs, dtype=np.float32), 0)
    with chainer.no_backprop_mode():
        q = Q(obs).data[0]
    return int(xp.argmax(q))


def mean_clipped_loss(y, t):
    # Add an axis because F.huber_loss only accepts arrays with ndim >= 2
    y = F.expand_dims(y, axis=-1)
    t = F.expand_dims(t, axis=-1)
    return F.sum(F.huber_loss(y, t, 1.0)) / y.shape[0]


def update(Q, target_Q, opt, samples, gamma=0.99, target_type='double_dqn'):
    xp = Q.xp
    s = xp.asarray([sample[0] for sample in samples], dtype=np.float32)
    a = xp.asarray([sample[1] for sample in samples], dtype=np.int32)
    r = xp.asarray([sample[2] for sample in samples], dtype=np.float32)
    done = xp.asarray([sample[3] for sample in samples], dtype=np.float32)
    s_next = xp.asarray([sample[4] for sample in samples], dtype=np.float32)
    # Predicted values: Q(s,a)
    y = F.select_item(Q(s), a)
    # Target values: r + gamma * max_b Q(s',b)
    with chainer.no_backprop_mode():
        if target_type == 'dqn':
            t = r + gamma * (1 - done) * F.max(target_Q(s_next), axis=1)
        elif target_type == 'double_dqn':
            t = r + gamma * (1 - done) * F.select_item(
                target_Q(s_next), F.argmax(Q(s_next), axis=1))
        else:
            raise ValueError('Unsupported target_type: {}'.format(target_type))
    loss = mean_clipped_loss(y, t)
    Q.cleargrads()
    loss.backward()
    opt.update()


def main():

    parser = argparse.ArgumentParser(description='Chainer example: DRL(DQN)')
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Name of the OpenAI Gym environment to play')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of transitions in each mini-batch')
    parser.add_argument('--episodes', '-e', type=int, default=1000,
                        help='Number of episodes to run')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='dqn_result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    parser.add_argument('--target-type', type=str, default='dqn',
                        help='Target type', choices=['dqn', 'double_dqn'])
    parser.add_argument('--reward-scale', type=float, default=1e-2,
                        help='Reward scale factor')
    parser.add_argument('--replay-start-size', type=int, default=500,
                        help='Number of steps after which replay is started')
    parser.add_argument('--steps-to-decay-epsilon', type=int, default=5000,
                        help='Number of steps used to linearly decay epsilon')
    parser.add_argument('--min-epsilon', type=float, default=0.01,
                        help='Minimum value of epsilon')
    parser.add_argument('--target-update-freq', type=int, default=100,
                        help='Frequency of target network update')
    parser.add_argument('--record', action='store_true', default=True,
                        help='Record performance')
    parser.add_argument('--no-record', action='store_false', dest='record')
    args = parser.parse_args()

    # Initialize an environment
    env = gym.make(args.env)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    ndim_obs = env.observation_space.low.size
    n_actions = env.action_space.n
    if args.record:
        env.monitor.start(args.out, force=True)

    # Initialize variables
    D = collections.deque(maxlen=10 ** 6)
    Rs = collections.deque(maxlen=100)
    step = 0

    # Initialize a model and its optimizer
    Q = QFunction(ndim_obs, n_actions, n_units=args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        Q.to_gpu(args.gpu)
    target_Q = copy.deepcopy(Q)
    opt = optimizers.Adam(eps=1e-2)
    opt.setup(Q)

    for episode in range(args.episodes):

        obs = env.reset()
        done = False
        R = 0.0
        t = 0

        while not done and t < env.spec.timestep_limit:

            # Epsilon is linearly decayed
            epsilon = 1.0 if len(D) < args.replay_start_size else \
                max(args.min_epsilon,
                    np.interp(
                        step,
                        [0, args.steps_to_decay_epsilon],
                        [1.0, args.min_epsilon]))

            # Select an action epsilon-greedily
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = get_greedy_action(Q, obs)

            # Execute an action
            new_obs, r, done, _ = env.step(a)
            R += r

            # Store a transition
            D.append((obs, a, r * args.reward_scale, done, new_obs))
            obs = new_obs

            # Sample a random minibatch of transitions and replay
            if len(D) >= args.replay_start_size:
                samples = random.sample(D, args.batchsize)
                update(Q, target_Q, opt, samples, target_type=args.target_type)

            # Update the target network
            if step % args.target_update_freq == 0:
                target_Q = copy.deepcopy(Q)

            step += 1
            t += 1

        Rs.append(R)
        average_R = np.mean(Rs)
        print('episode: {} step: {} R:{} average_R:{}'.format(
              episode, step, R, average_R))

    if args.record:
        env.monitor.close()


if __name__ == '__main__':
    main()
