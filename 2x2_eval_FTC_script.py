import os 
import csv
import pandas as pd
import numpy as np
import random 

import ray 
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms import ppo 
from ray.rllib.algorithms.algorithm import Algorithm

from environment.reward_functions import combined_reward_function_factory 
from environment.observation import EntireObservationFunction

from utils.environment_creator import par_pz_env_2x2_creator, par_pz_env_2x2_creator_without_outOfBoundsWrapper, init_env_params_2x2 
from utils.data_exporter import save_data_under_path

from torch.utils.tensorboard import SummaryWriter

from pprint import pprint

# -------------------- CONSTANTS -------------------------
ENV_NAME = "2x2grid"
NUM_EVAL_ITER = 1000
SIM_NUM_SECONDS = NUM_EVAL_ITER*5
SUMO_SEEDS = [39, 83, 49, 51, 74]
CSV_FILE_NAME = "eval_metrics.csv"

CHECKPOINT_DIR_NAMES = ["PPO_2024-04-18_12_59__alpha_0",
                        "PPO_2024-04-19_19_25__alpha_0.25",
                        "PPO_2024-04-20_16_34__alpha_0.5",
                        "PPO_2024-04-22_20_23__alpha_0.75",
                        "PPO_2024-04-22_08_37__alpha_1"]

alphas_to_test = [0, 0,25, 0.5, 0.75, 1]
alpha_checkpoint_zipped = list(zip(alphas_to_test,CHECKPOINT_DIR_NAMES))

# ----------- FOR PARTIAL TESTING ---------
PARTIAL_SUMO_SEEDS = [39]

for alpha, checkpoint_dir_name in alpha_checkpoint_zipped: 
    
    initial_path_to_store = f"reward_experiments/2x2grid/EVALUATION/{checkpoint_dir_name}"
    
    combined_reward_fn = combined_reward_function_factory(alpha)

    for sumo_seed in PARTIAL_SUMO_SEEDS:
        
        path_to_store =  os.path.abspath(os.path.join(initial_path_to_store,
                                                      "fixed_tc",
                                                      f"SEED_{sumo_seed}"))
        
        os.makedirs(path_to_store, exist_ok=True)

        csv_metrics_path = os.path.join(path_to_store, CSV_FILE_NAME)
        tb_log_dir = path_to_store

        env_params_eval = init_env_params_2x2(reward_function=combined_reward_fn,
                                              observation_function=EntireObservationFunction,
                                              num_seconds=SIM_NUM_SECONDS,
                                              sumo_seed=sumo_seed, 
                                              render=True,
                                              fixed_ts=True)


        # ------------------ SAVE DATA AND STORE ------------------

        eval_config_info = {"checkpoint_dir_name_to_evaluate": checkpoint_dir_name,
                            "number_eval_iterations": NUM_EVAL_ITER,
                            "congestion_coeff": alpha,
                            "evaluation_environment_args" : env_params_eval}
        try: 
            save_data_under_path(eval_config_info,
                                path_to_store,
                                "evaluation_info.json")
        except Exception as e:
            raise RuntimeError(f"An error occurred while saving evaluation info: {e}") from e

        # -----------BEGIN TESTING BASED ON FIXED TIME CONTROL--------

        rllib_pz_env = ParallelPettingZooEnv(
            par_pz_env_2x2_creator_without_outOfBoundsWrapper(env_params=env_params_eval,
                                                              single_eval_mode=True,
                                                              csv_path = csv_metrics_path,
                                                              tb_log_dir=tb_log_dir))

        agent_ids = rllib_pz_env.get_agent_ids()
        reward = {agent_id:0 for agent_id in agent_ids} 
        obs, info = rllib_pz_env.reset() 

        # this is dummy actions just to pass in so no exceptions are raised
        actions = {'1': 1, '2': 1, '5': 1, '6': 1}

        for i in range(NUM_EVAL_ITER):
            
            observations, rewards, terminations, truncations, infos = rllib_pz_env.step(actions)

            for agent_id in agent_ids:
                reward[agent_id] += rewards[agent_id]

        rllib_pz_env.close()
