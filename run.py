#!/usr/bin/env python
import gym
import numpy as np
import tensorflow as tf

import ddpg
import experiment
import filter_env

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('upload', False, 'upload to gym (requires evironment variable OPENAI_GYM_API_KEY)')
flags.DEFINE_string('env', '', 'gym environment')
flags.DEFINE_integer('train', 10, 'training episodes between tests. use 0 for test run only')
flags.DEFINE_integer('test', 1, 'testing episodes between training')
flags.DEFINE_integer('tmax', 10000, 'maximum timesteps per episode')
flags.DEFINE_bool('random', False, 'use random agent')
flags.DEFINE_bool('tot', False, 'train on test data')
flags.DEFINE_integer('total', 1000000, 'total episodes')
flags.DEFINE_float('monitor', .05, 'probability of monitoring a test episode')

VERSION = 'DDPG-v0'
GYM_ALGO_ID = 'alg_TmtzkcfSauZoBF97o9aQ'

if FLAGS.random:
    FLAGS.train = 0


class Experiment:

    def run(self):
        self.t_train = 0
        self.t_test = 0

        # create filtered environment
        self.env = filter_env.makeFilteredEnv(gym.make(FLAGS.env))
        # self.env = gym.make(FLAGS.env)

        self.env.monitor.start(FLAGS.outdir + '/monitor/', video_callable=lambda _: False)
        gym.logger.setLevel(gym.logging.WARNING)

        dimO = self.env.observation_space.shape
        dimA = self.env.action_space.shape
        print(dimO, dimA)

        import pprint
        pprint.pprint(self.env.spec.__dict__, width=1)

        self.agent = ddpg.Agent(dimO=dimO, dimA=dimA)

        simplelog = open(FLAGS.outdir + '/log.txt', 'w')

        # main loop
        while self.t_train < FLAGS.total:

            # test
            T = self.t_test
            R = []
            while self.t_test - T < FLAGS.test:
                R.append(self.run_episode(test=True, monitor=np.random.binomial(1, FLAGS.monitor)))
                self.t_test += 1
            avr = np.mean(R)
            print('Average test return\t{} after {} episodes of training'.format(avr, self.t_train))
            print >> simplelog, "%d\t%.2f" % (self.t_train, avr)

            # evaluate required number of episodes for gym and end training when above threshold
            if self.env.spec.reward_threshold is not None and avr > self.env.spec.reward_threshold:
                avr = np.mean([self.run_episode(test=True) for _ in range(self.env.spec.trials)])
                if avr > self.env.spec.reward_threshold:
                    break

            # train
            T = self.t_train
            R = []
            while self.t_train - T < FLAGS.train:
                R.append(self.run_episode(test=False))
                self.t_train += 1
            avr = np.mean(R)
            print('Average training return\t{} after {} episodes of training'.format(avr, self.t_train))

        self.env.monitor.close()
        # upload results
        if FLAGS.upload:
            gym.upload(FLAGS.outdir + "/monitor", algorithm_id=GYM_ALGO_ID)

    def run_episode(self, test=True, monitor=False):
        self.env.monitor.configure(lambda _: test and monitor)
        observation = self.env.reset()
        self.agent.reset(observation)
        R = 0  # return
        t = 1
        term = False
        while not term:
            # self.env.render(mode='human')

            if FLAGS.random:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(test=test)

            observation, reward, term, info = self.env.step(action)
            term = (t >= FLAGS.tmax) or term

            r_f = self.env.filter_reward(reward)
            self.agent.observe(r_f, term, observation, test=test and not FLAGS.tot)

            R += reward
            t += 1

        return R


def main():
    Experiment().run()

if __name__ == '__main__':
    experiment.run(main)
