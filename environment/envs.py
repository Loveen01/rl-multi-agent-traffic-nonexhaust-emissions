from collections import Counter
import csv
import os
import time
from typing import List, Optional, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from gymnasium.spaces import Discrete
from gymnasium.utils import EzPickle
from pettingzoo.utils import agent_selector

from sumo_rl import SumoEnvironment
from sumo_rl.environment.env import SumoEnvironmentPZ, parallel_env
import traci

import sys 

sys.path.append('/Users/loveen/Desktop/Masters project/rl-multi-agent-traffic-nonexhaust-emissions')
from environment.helper_functions import get_total_waiting_time, get_tyre_pm


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

# This file was taken from repo, https://github.com/case547/masters-proj/blob/master/envs.py, and has been altered to suit needs 

# this class ensures that we count rewards, instead of bypassing that delta_time period, it counts the rewards during delta time, 
# and returns this reward_hold dict as the computed rewards 
# overrides step(), _run_steps() and _compute_rewards()
class SumoEnvironmentCountAllRewards(SumoEnvironment):
    """Environment that counts rewards every sumo_step.
    
    Because delta_time != 1, the reward given to the agent(s) every
    step() is the sum of the last delta_time rewards generated by SUMO.
    """
    def __init__(self, eval=False, csv_path:Optional[str]=None, tb_log_dir:Optional[str]=None, **kwargs):
        # Call the parent constructor
        super().__init__(**kwargs)

        self.eval = eval
        self.csv_path = csv_path
        self.tb_log_dir = tb_log_dir

        if tb_log_dir:
            self.tb_writer = SummaryWriter(tb_log_dir)  # prep TensorBoard

        # Initialise cumulative counters
        self.tyre_pm_system = 0
        self.tyre_pm_agents = 0
        self.arrived_so_far = 0
        
        # Get traffic lights and lanes they control -> 
        self.controlled_lanes = [] # lo this will end up being an array of num_lanes controlled for each traffic signal. wierd how its a list, should be dict to preserve key value
        for ts in self.traffic_signals.values():
            self.controlled_lanes += ts.lanes

        self.max_dist = 200

        if csv_path:
            with open(csv_path, "w", newline="") as f:
                csv_writer = csv.writer(f, lineterminator='\n')
                headers = (["sim_time", "arrived_num", "sys_tyre_pm", "sys_stopped",
                            "sys_total_wait", "sys_avg_wait", "sys_avg_speed",
                            "agents_tyre_pm", "agents_stopped", "agents_total_delay", "agents_total_wait",
                            "agents_avg_delay", "agents_avg_wait", "agents_avg_speed"])
                csv_writer.writerow(headers)

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            "system_abs_accel": get_tyre_pm()
        }
    
    def _get_per_agent_info(self):
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        accumulated_waiting_time = [
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.ts_ids
        ]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        abs_accelerations = [get_tyre_pm(ts) for ts in self.traffic_signals.values()]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
            info[f"{ts}_abs_accel"] = abs_accelerations[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info
    
    def _sumo_step(self):
        self.sumo.simulationStep() # lo. this is solely offered in the original function 

        if self.eval:
            # SYSTEM
            vehicles = self.sumo.vehicle.getIDList() # get all vehicles in env
            speeds = [self.sumo.vehicle.getSpeed(veh) for veh in vehicles] # [speedv1 speedv2]
            waiting_times = [self.sumo.vehicle.getWaitingTime(veh) for veh in vehicles] # [waitv1 waitv2]

            system_tyre_pm = get_tyre_pm()  # lo. tyre emmissions for all vehicles in simulation...
            arrived_num = self.sumo.simulation.getArrivedNumber() 
            self.tyre_pm_system += system_tyre_pm # lo. append to cumulative...
            self.arrived_so_far += arrived_num # lo. append to cumulative counter...
            
            system_stats = {
                "total_stopped": sum(int(speed < 0.1) for speed in speeds),
                "total_waiting_time": sum(waiting_times),
                "mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
                "mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            }
            
            # AGENTS
            observable_vehs = []
            for lane in self.controlled_lanes: # lo. 
                observable_vehs += self.get_observable_vehs(lane)

            agent_speeds = [self.sumo.vehicle.getSpeed(veh) for veh in observable_vehs]
            agent_waiting_times = [self.sumo.vehicle.getWaitingTime(veh) for veh in observable_vehs]
            agent_accum_waits = [self.sumo.vehicle.getAccumulatedWaitingTime(veh) for veh in observable_vehs]
            
            agents_tyre_pm = sum(abs(self.sumo.vehicle.getAcceleration(veh)) for veh in observable_vehs)
            self.tyre_pm_agents += agents_tyre_pm
            
            agent_stats = {
                # ts: TrafficSignal
                "total_stopped": sum(int(speed < 0.1) for speed in agent_speeds),
                "total_accum_wait": sum(agent_accum_waits),
                "total_waiting_time": sum(agent_waiting_times),
                
                "avg_accum_wait": 0.0 if len(observable_vehs) == 0 else np.mean(agent_accum_waits),
                "avg_waiting_time": 0.0 if len(observable_vehs) == 0 else np.mean(agent_waiting_times),
                "avg_speed": 0.0 if len(observable_vehs) == 0 else np.mean(agent_speeds),
            }
            
            # Log to CSV
            if self.csv_path:
                with open(self.csv_path, "a", newline="", ) as f:
                    csv_writer = csv.writer(f, lineterminator='\n')
                    data = ([self.sim_step, arrived_num, system_tyre_pm]
                            + list(system_stats.values())
                            + [agents_tyre_pm]
                            + list(agent_stats.values()))
                    
                    csv_writer.writerow(data)

            # Log to TensorBoard
            if hasattr(self, "tb_writer"):
                # System
                self.tb_writer.add_scalar("world/arrived_so_far", self.arrived_so_far, self.sim_step)
                self.tb_writer.add_scalar("world/tyre_pm_cumulative", self.tyre_pm_system, self.sim_step)

                for stat, val in system_stats.items():
                    self.tb_writer.add_scalar(f"world/{stat}", val, self.sim_step)

                # Agents
                self.tb_writer.add_scalar("agents/tyre_pm_cumulative", self.tyre_pm_agents, self.sim_step)

                for stat, val in agent_stats.items():
                    self.tb_writer.add_scalar(f"agents/{stat}", val, self.sim_step)

    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            # Rewards for the sumo steps between every env step
            self.reward_hold = Counter({ts: 0 for ts in self.ts_ids})
            for _ in range(self.delta_time):
                self._sumo_step()

                r = {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids}
                self.reward_hold.update(r)  # add r to reward_hold Counter
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self._compute_info()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info
    
    def _sumo_step(self):
        self.sumo.simulationStep() # lo. this is solely offered in the original function 

        if self.eval:
            # SYSTEM
            vehicles = self.sumo.vehicle.getIDList() # get all vehicles in env
            speeds = [self.sumo.vehicle.getSpeed(veh) for veh in vehicles] # [speedv1 speedv2]
            waiting_times = [self.sumo.vehicle.getWaitingTime(veh) for veh in vehicles] # [waitv1 waitv2]

            system_tyre_pm = get_tyre_pm()  # lo. tyre emmissions for all vehicles in simulation...
            arrived_num = self.sumo.simulation.getArrivedNumber() # 
            self.tyre_pm_system += system_tyre_pm # lo. append to cumulative...
            self.arrived_so_far += arrived_num # lo. append to cumulative counter...
            
            system_stats = {
                "total_stopped": sum(int(speed < 0.1) for speed in speeds),
                "total_waiting_time": sum(waiting_times),
                "mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
                "mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            }
            
            # AGENTS
            observable_vehs = []
            for lane in self.controlled_lanes: # lo. 
                observable_vehs += self.get_observable_vehs(lane)

            agent_speeds = [self.sumo.vehicle.getSpeed(veh) for veh in observable_vehs]
            agent_waiting_times = [self.sumo.vehicle.getWaitingTime(veh) for veh in observable_vehs]
            agent_accum_waits = [self.sumo.vehicle.getAccumulatedWaitingTime(veh) for veh in observable_vehs]
            
            agents_tyre_pm = sum(abs(self.sumo.vehicle.getAcceleration(veh)) for veh in observable_vehs)
            self.tyre_pm_agents += agents_tyre_pm
            
            agent_stats = {
                # ts: TrafficSignal
                "total_stopped": sum(int(speed < 0.1) for speed in agent_speeds),
                "total_accum_wait": sum(agent_accum_waits),
                "total_waiting_time": sum(agent_waiting_times),
                
                "avg_accum_wait": 0.0 if len(observable_vehs) == 0 else np.mean(agent_accum_waits),
                "avg_waiting_time": 0.0 if len(observable_vehs) == 0 else np.mean(agent_waiting_times),
                "avg_speed": 0.0 if len(observable_vehs) == 0 else np.mean(agent_speeds),
            }
            
            # Log to CSV
            if self.csv_path:
                with open(self.csv_path, "a", newline="") as f:
                    csv_writer = csv.writer(f)
                    data = ([self.sim_step, arrived_num, system_tyre_pm]
                            + list(system_stats.values())
                            + [agents_tyre_pm]
                            + list(agent_stats.values()))
                    
                    csv_writer.writerow(data)

            # Log to TensorBoard
            if hasattr(self, "tb_writer"):
                # System
                self.tb_writer.add_scalar("world/arrived_so_far", self.arrived_so_far, self.sim_step)
                self.tb_writer.add_scalar("world/tyre_pm_cumulative", self.tyre_pm_system, self.sim_step)

                for stat, val in system_stats.items():
                    self.tb_writer.add_scalar(f"world/{stat}", val, self.sim_step)

                # Agents
                self.tb_writer.add_scalar("agents/tyre_pm_cumulative", self.tyre_pm_agents, self.sim_step)

                for stat, val in agent_stats.items():
                    self.tb_writer.add_scalar(f"agents/{stat}", val, self.sim_step)

    def _run_steps(self): # this func is also called in step() function 
        # Rewards for the sumo steps between every env step
        self.reward_hold = Counter({ts: 0 for ts in self.ts_ids})
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            r = {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids}
            self.reward_hold.update(r)  # add r to reward_hold Counter

            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _compute_rewards(self): # this func is called in step()
        self.rewards.update(
            {ts: self.reward_hold[ts] for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act} # same thing as returning rewards_hold ... why not return rewards_hold?

    def _compute_info(self):
        info = {"__common__": {"step": self.sim_step}}
        per_agent_info = self._get_per_agent_info()

        for agent_id in self.ts_ids:
            agent_info = {}

            for k, v in per_agent_info.items():
                if k.startswith(agent_id):
                    agent_info[k.split("_")[-1]] = v

            # Add tyre PM
            agent_info["tyre_pm"] = get_tyre_pm(self.traffic_signals[agent_id])

            info.update({agent_id: agent_info})

        return info

    def get_observable_vehs(self, lane) -> List[str]:
        """Remove undetectable vehicles from a lane."""
        detectable = []
        for vehicle in self.sumo.lane.getLastStepVehicleIDs(lane): # Returns the ids of the vehicles for the last time step on the given lane
            path = self.sumo.vehicle.getNextTLS(vehicle) # Return list of upcoming traffic lights [(tlsID, tlsIndex, distance, state), ...]
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= self.max_dist:  # Detectors have a max range. if distance from vehicle to traffic light is < 200, vehicle is detectable from traffic light
                    detectable.append(vehicle)
        return detectable

# class same as SumoEnvironmentPZ, (as it inherits from sumoEnvPZ), BUT THIS TIME, 
# instead of self.env = SumoEnvironment(), its self.env = SumoEnvironmentCountAllRewards()
class SumoEnvironmentPZCountAllRewards(SumoEnvironmentPZ):
    """A wrapper for `CountAllRewardsEnv` that adds additional metrics every SUMO step"""
    def __init__(self, eval=False, csv_path: Optional[str] = None, tb_log_dir: Optional[str] = None, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEnvironmentCountAllRewards(**self._kwargs, eval=eval, csv_path=csv_path, tb_log_dir=tb_log_dir)  # instead of SumoEnvironment. CountAllRewardsEnv is subclass of SumoEnv  

        self.agents = self.env.ts_ids  # do you really need to redefine all the attributes, they are already inherited from SumoEnvPz?
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    # @property
    # def action_space(self):
    #     """Return the biggest action space of the traffic signal agents."""
    #     spaces = [self.action_spaces[ts] for ts in self.env.traffic_signals]
    #     max_n = max([space.n for space in spaces])
    #     return Discrete(max_n)

    # def close(self):
    #     """Close the environment and stop the SUMO simulation."""
    #     if self.env.sumo is None:
    #         return

    #     if not LIBSUMO:
    #         traci.switch(self.env.label)
    #     traci.close()

    #     # Help completely release SUMO port between episodes to address
    #     # "Unable to create listening socket: Address already in use" error
    #     time.sleep(2)

    #     if self.env.disp is not None:
    #         self.env.disp.stop()
    #         self.env.disp = None

    #     self.env.sumo = None