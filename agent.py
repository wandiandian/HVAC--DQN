import numpy as np
from approximator import NetApproximator
import random
from tqdm import tqdm


class Transition(object):
    def __init__(self, s0, a0, reward: float, is_done: bool, s1):
        self.data = [s0, a0, reward, is_done, s1]

    # 如果一个类想被用于for ... in循环，类似list或tuple那样，就必须实现一个__iter__()方法，该方法返回一个迭代对象，
    def __iter__(self):
        return iter(self.data)  # iter() 函数用来生成迭代器。

    # 当使用print输出对象的时候，只要自己定义了__str__(self)方法，那么就会打印从在这个方法中return的数据
    # 在python中方法名如果是__xxxx__()的，那么就有特殊的功能，因此叫做“魔法”方法
    def __str__(self):
        return "s:{0:<3} a:{1:<3} r:{2:<4} is_end:{3:<5} s1:{4:<3}". \
            format(self.data[0], self.data[1], self.data[2],
                   self.data[3], self.data[4])

    @property
    def s0(self): return self.data[0]

    @property
    def a0(self): return self.data[1]

    @property
    def reward(self): return self.data[2]

    @property
    def is_done(self): return self.data[3]

    @property
    def s1(self): return self.data[4]


class Episode(object):
    def __init__(self, e_id: int = 0) -> None:  # ->常常出现在python函数定义的函数名后面，为函数添加元数据,描述函数的返回类型
        self.total_reward = 0  # 总的获得的奖励
        self.trans_list = []  # 状态转移列表
        self.name = str(e_id)  # 可以给Episode起个名字："成功闯关,黯然失败？"

    def push(self, trans: Transition) -> float:
        self.trans_list.append(trans)
        self.total_reward += trans.reward  # 不计衰减的总奖励
        return self.total_reward

    @property
    def len(self):
        return len(self.trans_list)

    def is_complete(self) -> bool:
        """check if an episode is an complete episode
        """
        if self.len == 0:
            return False
        return self.trans_list[self.len - 1].is_done

    def sample(self, batch_size=1):
        """随即产生一个trans
        """
        # 从list中随机取出k个元素
        return random.sample(self.trans_list, k=batch_size)


class Experience(object):
    """this class is used to record the whole experience of an agent organized
    by an episode list. agent can randomly sample transitions or episodes from
    its experience.
    """

    def __init__(self, capacity: int = 20000):
        self.capacity = capacity  # 容量：指的是trans总数量
        self.episodes = []  # episode列表
        self.next_id = 0  # 下一个episode的Id
        self.total_trans = 0  # 总的状态转换数量

    def __len__(self):
        return self.len

    @property
    def len(self):
        return len(self.episodes)

    def _remove(self, index=0):
        """扔掉一个Episode，默认第一个。
           remove an episode, defautly the first one.
           args:
               the index of the episode to remove
           return:
               if exists return the episode else return None
        """
        if index > self.len - 1:
            raise (Exception("invalid index"))
        if self.len > 0:
            episode = self.episodes[index]
            self.episodes.remove(episode)
            self.total_trans -= episode.len
            return episode
        else:
            return None

    def _remove_first(self):
        self._remove(index=0)

    def push(self, trans):
        """压入一个状态转换
        """
        if self.capacity <= 0:
            return
        while self.total_trans >= self.capacity:  # 可能会有空episode吗？
            self._remove_first()
        if self.len == 0 or self.episodes[self.len - 1].is_complete():
            cur_episode = Episode(self.next_id)  # 创建一个空的Episode
            self.next_id += 1
            self.episodes.append(cur_episode)
        else:
            cur_episode = self.episodes[self.len - 1]
        self.total_trans += 1
        return cur_episode.push(trans)  # return  total reward of an episode

    def sample(self, batch_size=1):  # sample transition
        """randomly sample some transitions from agent's experience.abs
        随机获取一定数量的状态转化对象Transition
        args:
            number of transitions need to be sampled
        return:
            list of Transition.
        """
        sample_trans = []
        for _ in range(batch_size):
            index = int(random.random() * self.len)
            sample_trans += self.episodes[index].sample()
        return sample_trans

    def sample_episode(self, episode_num=1):  # sample episode
        """随机获取一定数量完整的Episode
        """
        return random.sample(self.episodes, k=episode_num)

    @property
    def last_episode(self):
        if self.len > 0:
            return self.episodes[self.len - 1]
        return None


class DQNAgent:
    """使用近似的价值函数实现的Q学习个体
    """

    def __init__(self,
                 env,
                 capacity=20000,
                 hidden_dim: int = 32,
                 batch_size=100,
                 epochs=2):
        if env is None:
            raise Exception("agent should have an environment")
        # super(DQNAgent, self).__init__(env, capacity)
        self.env = env  # 建立对环境对象的引用
        self.S = None
        self.A = None
        self.experience = Experience(capacity=capacity)
        self.input_dim = 7
        self.output_dim = 16
        self.hidden_dim = hidden_dim
        # 行为网络，该网络用来计算产生行为，以及对应的Q值，每次更新
        self.behavior_Q = NetApproximator(input_dim=self.input_dim,
                                          output_dim=self.output_dim,
                                          hidden_dim=self.hidden_dim)
        self.target_Q = self.behavior_Q.clone()  # 计算价值目标的Q，不定期更新

        self.batch_size = batch_size  # 批学习一次状态转换数量
        self.epochs = epochs  # 统一批状态转换神经网络学习的次数 approximator.py 62行
        return

    def _update_target_Q(self):
        """将更新策略的Q网络(连带其参数)复制给输出目标Q值的网络
        """
        self.target_Q = self.behavior_Q.clone()  # 更新计算价值目标的Q网络

    # def policy(self, A, s, Q=None, epsilon=None):
    def policy(self, s, epsilon=None):
        """依据更新策略的价值函数(网络)产生一个行为
        """
        Q_s = self.behavior_Q(s)
        rand_value = random.random()
        if epsilon is not None and rand_value < epsilon:
            # return self.env.action_space.sample()  # discrete.py19行，因该是系统文件
            return int(np.random.rand() * 16)
        else:
            return int(np.argmax(Q_s))

    def _decayed_epsilon(self, cur_episode: int,
                         min_epsilon: float,
                         max_epsilon: float,
                         target_episode: int) -> float:  # 该episode及以后的episode均使用min_epsilon
        """获得一个在一定范围内的epsilon
        """
        slope = (min_epsilon - max_epsilon) / target_episode
        intercept = max_epsilon
        return max(min_epsilon, slope * cur_episode + intercept)

    # 进行max_episode_num轮学习
    def learning(self, lambda_=0.9, epsilon=None, decaying_epsilon=True, gamma=0.9,
                 alpha=0.1, max_episode_num=800, display=False, min_epsilon=1e-1, min_epsilon_ratio=0.8):
        total_time, episode_reward, num_episode = 0, 0, 0
        total_times, episode_rewards, num_episodes = [], [], []
        for i in tqdm(range(max_episode_num)):
            if epsilon is None:
                epsilon = 1e-10
            elif decaying_epsilon:
                epsilon = max(1 - (1/(max_episode_num-5)) * i, 0)
                # epsilon = self._decayed_epsilon(cur_episode=num_episode + 1,
                #                                 min_epsilon=min_epsilon,
                #                                 max_epsilon=1.0,
                #                                 target_episode=int(max_episode_num * min_epsilon_ratio))
            # 进行一轮学习
            time_in_episode, episode_reward = self.learning_method(lambda_=lambda_,
                                                                   gamma=gamma, alpha=alpha, epsilon=epsilon,
                                                                   display=display)  # agent 224
            total_time += time_in_episode
            num_episode += 1
            total_times.append(total_time)
            episode_rewards.append(episode_reward)
            num_episodes.append(num_episode)
        # self.experience.last_episode.print_detail()
        return total_times, episode_rewards, num_episodes

    def learning_method(self, lambda_=None, gamma=0.9, alpha=0.1, epsilon=1e-5,
                        display=False):
        self.state = self.env.reset()  # 应该是继承的Agent的值
        # s0 = self.state
        if display:
            self.env.render()
        time_in_episode, total_reward = 0, 0
        is_done = False
        loss = 0
        while not is_done:
            # add code here
            s0 = self.state  # 应该是继承的Agent的值
            # policy函数中会进行神经网络前向计算从而归一化，传参s0_copy避免s0被归一化
            a0 = self.policy(s0, epsilon)
            s1, r1, is_done, info, total_reward = self.act(a0)  # 此函数将s1赋给s0了
            if display:
                self.env.render()

            if self.total_trans > self.batch_size:
                loss += self._learn_from_memory(gamma, alpha)
            # s0 = s1
            time_in_episode += 10
            is_done = bool(time_in_episode == 960)  # 一天24小时1440分钟
        loss /= time_in_episode
        if display:
            print("epsilon:{:3.2f},loss:{:3.2f},{}".format(epsilon, loss, self.experience.last_episode))
        return time_in_episode, total_reward

    def act(self, a0):
        s0 = self.state  # 可能在puckworld119行首次初始化或者下面的233行
        s1, r1, is_done, info = self.env.step(a0)  # puckworld 70行
        # TODO add extra code here
        trans = Transition(s0, a0, r1, is_done, s1)
        total_reward = self.experience.push(trans)
        self.state = s1
        return s1, r1, is_done, info, total_reward

    def _learn_from_memory(self, gamma, learning_rate):
        trans_pieces = self.sample(self.batch_size)  # 随机获取记忆里的Transmition继承的agent方法
        states_0 = np.vstack([x.s0 for x in trans_pieces])
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])

        X_batch = states_0  # 128*6
        y_batch = self.behavior_Q(states_0)  # 128*5 得到numpy格式的结果
        # 0，表示每一列的最大值的索引，axis=1表示每一行的最大值的索引#target_Q(states_1)行和列表示什么？
        u=self.behavior_Q(states_1)
        i=np.max(self.behavior_Q(states_1),axis=1)
        n=gamma * np.max(self.behavior_Q(states_1), axis=1)
        Q_target = reward_1 + gamma * np.max(self.behavior_Q(states_1), axis=1) * \
                   (~ is_done)  # 128*1 is_done则Q_target==reward_1
        # for i in range(len(Q_target)):
        #     if Q_target[i] < -1:
        #         Q_target[i] = -1
        # switch this on will make DQN to DDQN
        # 行为a'从行为价值网络中得到
        # a_prime = np.argmax(self.behavior_Q(states_1), axis=1).reshape(-1)#reshape(-1)拉成一行
        # (s',a')的价值从目标价值网络中得到
        # Q_states_1 = self.target_Q(states_1)
        # temp_Q = Q_states_1[np.arange(len(Q_states_1)), a_prime]
        # (s,a)的目标价值根据贝尔曼方程得到
        # Q_target = reward_1 + gamma * temp_Q * (~ is_done) # is_done则Q_target==reward_1
        # end of DDQN part
        # 函数返回一个有终点和起点的固定步长的排列 一个参数时，参数值为终点，起点取默认值0，步长取默认值1 不包括终点
        # x = np.arange(len(X_batch))  # 0-127
        # y = len(X_batch)  # 128
        y_batch[np.arange(len(X_batch)), actions_0] = Q_target  # 全是ndarry
        # y_batch是128*5的二维矩阵，np.arange(len(X_batch))是y_batch第一维索引，actions_0是第二维索引，他俩组成（x，y）是Q_target插入位置
        # loss is a torch Variable with size of 1
        loss = self.behavior_Q.fit(x=X_batch,
                                   y=y_batch,
                                   learning_rate=learning_rate,
                                   epochs=self.epochs)

        # mean_loss = loss.sum().data[0] / self.batch_size
        mean_loss = loss.sum().item() / self.batch_size  # .item()转成python数字
        self._update_target_Q()
        return mean_loss

    def sample(self, batch_size=64):
        """随机取样
        """
        return self.experience.sample(batch_size)

    @property
    def total_trans(self):
        """得到Experience里记录的总的状态转换数量
        """
        return self.experience.total_trans
