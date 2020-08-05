import numpy as np
import math


class Building:
    """A simple Building Energy Model.

    Consisting of one thermal capacity and one resistance, this model is derived from the
    hourly dynamic model of the ISO 13790. It models heating and cooling energy demand only.

    Parameters:
        * heat_mass_capacity:           capacity of the building's heat mass [J/K]
        * heat_transmission:            heat transmission to the outside [W/K]
        * initial_building_temperature: building temperature at start time [℃]
        * initial_out_temperature:      outside temperature at start time [℃]
        * comfortable_temp_upper:       the upper bounder of  comfortable temperature [℃]
        * comfortable_temp_lower:       the lower bounder of  comfortable temperature [℃]
        * time_step_size:               [s]
    """

    def __init__(self, heat_mass_capacity, heat_transmission,
                 initial_building_temperature, initial_out_temperature,
                 comfortable_temp_upper, comfortable_temp_lower, time_step_size):
        self.__heat_mass_capacity = heat_mass_capacity
        self.__heat_transmission = heat_transmission
        self.in_temperature = initial_building_temperature
        self.out_temperature = initial_out_temperature
        self.comfortable_temp_lower = comfortable_temp_lower
        self.comfortable_temp_upper = comfortable_temp_upper
        self.__time_step_size = time_step_size
        self.reward = 0
        self.state = np.array([0, 0, 0, 0, 0, 0, 0])
        self.time = 0
        self.temperature = []
        self.price = 0
        self.ice = 10445
        self.timeone = 0
        self.timetwo = 0
        self.timethree = 0
        self.timefour = 0

    def step(self, action):
        """Performs building simulation for the next time step.

        Parameters:
            action:空调功率
        """
        if action == 0:
            hvac_power = 0
            ice_power = 0
        elif action == 1:
            hvac_power = 0
            ice_power = 400
        elif action == 2:
            hvac_power = 0
            ice_power = 800
        elif action == 3:
            hvac_power = 0
            ice_power = 1200
        elif action == 4:
            hvac_power = 400
            ice_power = 0
        elif action == 5:
            hvac_power = 400
            ice_power = 400
        elif action == 6:
            hvac_power = 400
            ice_power = 800
        elif action == 7:
            hvac_power = 400
            ice_power = 1200
        elif action == 8:
            hvac_power = 800
            ice_power = 0
        elif action == 9:
            hvac_power = 800
            ice_power = 400
        elif action == 10:
            hvac_power = 800
            ice_power = 800
        elif action == 11:
            hvac_power = 800
            ice_power = 1200
        elif action == 12:
            hvac_power = 1200
            ice_power = 0
        elif action == 13:
            hvac_power = 1200
            ice_power = 400
        elif action == 14:
            hvac_power = 1200
            ice_power = 800
        elif action == 15:
            hvac_power = 1200
            ice_power = 1200

        self.timeone, self.timetwo, self.timethree, self.timefour, in_temperature, out_temperature, self.ice = self.state
        # dt_by_cm = self.__time_step_size.total_seconds() / self.__heat_mass_capacity
        # hvac = heating_cooling_power * 0.000179
        # transmission = (outside_temperature - self.current_temperature) * 0.000041
        # return (self.current_temperature * (1 - dt_by_cm * self.__heat_transmission) +
        #         dt_by_cm * (heating_cooling_power + self.__heat_transmission * outside_temperature))
        # 下一时刻状态计算
        # hvac = dt_by_cm * heating_cooling_power
        # transmission = (out_temperature - in_temperature) \
        #                * dt_by_cm * self.__heat_transmission
        # next_temperature = in_temperature + hvac + transmission
        # next_temperature = in_temperature + heating_cooling_power * 0.01074 + \
        #                    (out_temperature - in_temperature) * 0.00246

        # 计算电价
        if (0 <= self.time < 120) or (240 <= self.time < 600) or (840 <= self.time < 960):
            self.price = 0.6907
        else:
            self.price = 1.2433

        # 计算剩余冰量
        self.ice = self.ice - ice_power / 6
        if self.ice < 0:
            ice_power = 0

        # 下一时刻室内温度
        next_temperature = in_temperature - (hvac_power + ice_power) * 0.002917 + \
                           (out_temperature - in_temperature) * 0.5

        self.temperature.append(next_temperature)

        # 奖励函数计算，记得变成负数
        # 电价(每10分钟)
        hp = 0.0001238 * hvac_power * hvac_power + 0.00932 * hvac_power + 176.1
        if -0.1 < hvac_power < 0.1:
            hp = 0
        if ice_power > 1:
            self.reward = (hp + 90) * self.price / 6 + (ice_power / 2.55 * 0.3315) / 6
        else:
            self.reward = hp * self.price / 6

        # 温度越界惩罚
        # self.reward = 0
        if next_temperature > self.comfortable_temp_upper:
            self.reward += 100 * (next_temperature - self.comfortable_temp_upper)
        elif next_temperature < self.comfortable_temp_lower:
            self.reward += 100 * (self.comfortable_temp_lower - next_temperature)
        # else:
        #     self.reward = 0

        # 余冰不足惩罚
        # if self.ice < 0:
        #     self.reward += abs(self.ice)

        # 奖励函数值取负
        self.reward = -self.reward

        # 下一时刻状态
        if 0 <= self.time < 60:
            self.timeone = 0
            self.timetwo = 0
            self.timethree = 0
            self.timefour = 0
        elif 60 <= self.time < 120:
            self.timeone = 0
            self.timetwo = 0
            self.timethree = 0
            self.timefour = 1
        elif 120 <= self.time < 180:
            self.timeone = 0
            self.timetwo = 0
            self.timethree = 1
            self.timefour = 0
        elif 180 <= self.time < 240:
            self.timeone = 0
            self.timetwo = 0
            self.timethree = 1
            self.timefour = 1
        elif 240 <= self.time < 300:
            self.timeone = 0
            self.timetwo = 1
            self.timethree = 0
            self.timefour = 0
        elif 300 <= self.time < 360:
            self.timeone = 0
            self.timetwo = 1
            self.timethree = 0
            self.timefour = 1
        elif 360 <= self.time < 420:
            self.timeone = 0
            self.timetwo = 1
            self.timethree = 1
            self.timefour = 0
        elif 420 <= self.time < 480:
            self.timeone = 0
            self.timetwo = 1
            self.timethree = 1
            self.timefour = 1
        elif 480 <= self.time < 540:
            self.timeone = 1
            self.timetwo = 0
            self.timethree = 0
            self.timefour = 0
        elif 540 <= self.time < 600:
            self.timeone = 1
            self.timetwo = 0
            self.timethree = 0
            self.timefour = 1
        elif 600 <= self.time < 660:
            self.timeone = 1
            self.timetwo = 0
            self.timethree = 1
            self.timefour = 0
        elif 660 <= self.time < 720:
            self.timeone = 1
            self.timetwo = 0
            self.timethree = 1
            self.timefour = 1
        elif 720 <= self.time < 780:
            self.timeone = 1
            self.timetwo = 1
            self.timethree = 0
            self.timefour = 0
        elif 780 <= self.time < 840:
            self.timeone = 1
            self.timetwo = 1
            self.timethree = 0
            self.timefour = 1
        elif 840 <= self.time < 900:
            self.timeone = 1
            self.timetwo = 1
            self.timethree = 1
            self.timefour = 0
        elif 900 <= self.time < 960:
            self.timeone = 1
            self.timetwo = 1
            self.timethree = 1
            self.timefour = 1

        self.time = (self.time + 10) % 960  # 从8点开始
        out_temperature = self.out_temperature + 4.0 * math.sin(math.pi / 12 * ((self.time + 480) / 60 - 10))
        self.state = np.array([self.timeone,
                               self.timetwo,
                               self.timethree,
                               self.timefour,
                               next_temperature,
                               out_temperature,
                               self.ice
                               ])
        done = bool(self.time == 0)
        return self.state, self.reward, done, {}

    # 分别初始化输入参数
    # self.timeone,
    # self.timetwo,
    # self.timethree,
    # self.timefour,
    # self.in_temperature,
    # self.out_temperature,
    # self.ice
    def reset(self):
        self.state = np.array([0,
                               0,
                               0,
                               0,
                               28.0,
                               28.0,  # 8点室外也是28℃
                               10445.0
                               ])
        return self.state
