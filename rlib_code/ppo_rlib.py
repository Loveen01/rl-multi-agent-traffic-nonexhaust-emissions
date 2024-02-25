import os 
import pathlib

import sumo_rl
from sumo_rl.environment.env import env, parallel_env, SumoEnvironment

import ray

from ray.tune import register_env
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.algorithms import ppo 

from environment.envs import RealMultiAgentSumoEnv
from environment.observation import Grid2x2ObservationFunction, EntireObservationFunction
from environment.reward_functions import combined_reward


os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
os.environ['RAY_PICKLE_VERBOSE_DEBUG'] = '1'


env_folder = "data/2x2grid"

env_name = "2x2grid"

multi_agent_env = parallel_env(    
        net_file = os.path.join(env_folder, "2x2.net.xml"),
        route_file = os.path.join(env_folder, "2x2.rou.xml"),
        reward_fn = combined_reward,
        observation_class = EntireObservationFunction, 
        out_csv_name="outputs/2x2grid/ppo", 
        num_seconds=1000,
        add_per_agent_info=True,
        add_system_info=True,
        single_agent=False)



seed = 4

env_params = {"net_file": os.path.join(env_folder, "2x2.net.xml"),
        "route_file": os.path.join(env_folder, "2x2.rou.xml"),
        "reward_fn": combined_reward,
        "observation_class": EntireObservationFunction, 
        "out_csv_name": "outputs/2x2grid/ppo", 
        "num_seconds": 1000,
        "add_per_agent_info": True,
        "add_system_info": True,
        "sumo_seed": seed,
        "single_agent": False}

ray.init()

multi_agent_par_env = RealMultiAgentSumoEnv(**env_params) # SUMO environment implementing PettingZoo API TODO: CHANGE NAME

rllib_compat_ppz_env = ParallelPettingZooEnv(multi_agent_par_env) # Wrap it to be a Parallel Petting Zoo env 

register_env(name=env_name, env_creator= lambda config : rllib_compat_ppz_env) # register env

config = (
        ppo.PPOConfig()
        .environment(env_name)
        # .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=1)
        # .multi_agent(
        #         policies=multi_agent_par_env.get_agent_ids(),
        #         policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        # )
)

config.build()
