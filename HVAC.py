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
        self.state = np.array([0, 0])
        self.time = 0
        self.temperature = []

    def step(self, action):
        """Performs building simulation for the next time step.

        Parameters:
            action:空调功率
        """
        if action == 0:
            heating_cooling_power = -3000
        if action == 1:
            heating_cooling_power = -2000
        if action == 2:
            heating_cooling_power = -1000
        if action == 3:
            heating_cooling_power = 0
        if action == 4:
            heating_cooling_power = 1000
        if action == 5:
            heating_cooling_power = 2000
        if action == 6:
            heating_cooling_power = 3000
        in_temperature, out_temperature = self.state
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
        next_temperature = in_temperature + heating_cooling_power * 0.000179 + \
                           (out_temperature - in_temperature) * 0.000041
        self.temperature.append(next_temperature)

        self.time = (self.time + 1) % 1440
        out_temperature = self.out_temperature + 3.85 * math.sin(math.pi / 12 * (self.time / 60 - 10))
        self.state = np.array([next_temperature,
                               out_temperature
                               ])
        # 奖励函数计算
        if next_temperature > self.comfortable_temp_upper:
            # /60是计算每分钟电费
            self.reward = (-abs(heating_cooling_power) / 60 / 1000 - 100 * (
                    next_temperature - self.comfortable_temp_upper)) / 1000
            # self.reward = - 1000 * (next_temperature - self.comfortable_temp_upper)
        elif next_temperature < self.comfortable_temp_lower:
            self.reward = (-abs(heating_cooling_power) / 60 / 1000 - 100 * (
                    self.comfortable_temp_lower - next_temperature)) / 1000
            # self.reward = - 1000 * (self.comfortable_temp_lower - next_temperature)
        else:
            self.reward = (-abs(heating_cooling_power) / 60 / 1000 - 100 * 0) / 1000
            # self.reward = 0
        done = bool(self.time == 0)
        return self.state, self.reward, done, {}

    def reset(self):
        self.state = np.array([self.in_temperature,
                               self.out_temperature
                               ])
        return self.state
