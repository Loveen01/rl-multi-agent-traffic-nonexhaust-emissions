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

from environment.reward_functions import combined_reward_function_factory_with_delta_wait_time 
from environment.observation import EntireObservationFunction

from utils.environment_creator import par_pz_env_2x2_creator, par_pz_env_2x2_creator_without_outOfBoundsWrapper, init_env_params_2x2 
from utils.data_exporter import save_data_under_path

from torch.utils.tensorboard import SummaryWriter

from pprint import pprint

# -------------------- CONSTANTS ---------------------------
NUM_ENV_STEPS = 1000
SIM_NUM_SECONDS = NUM_ENV_STEPS*5
SUMO_SEEDS = [39, 83, 49, 51, 74]
CSV_FILE_NAME = "eval_metrics.csv"
EXTRA_METRICS_CSV_FILE_NAME = "extra_metrics.csv"

# -------------------- ORIG VARIABLES IN EXPERIMENT -------------------------
ALPHAS_TO_TEST = [0, 0.25, 0.5, 0.75, 1]
CHECKPOINT_DIR_NAMES = ["PPO_2024-04-18_12_59__alpha_0",
                        "PPO_2024-04-19_19_25__alpha_0.25",
                        "PPO_2024-04-20_16_34__alpha_0.5",
                        "PPO_2024-04-22_20_23__alpha_0.75",
                        "PPO_2024-04-22_08_37__alpha_1"]

alpha_checkpoint_zipped = list(zip(ALPHAS_TO_TEST, CHECKPOINT_DIR_NAMES))

# ---------------- OVERRIDE FOR PARTIAL TESTING FOR SPECIFIC NEEDS ---------------

ALPHAS_TO_TEST = [0.25, 0.5, 0.75, 1]
CHECKPOINT_DIR_NAMES = ["PPO_2024-04-19_19_25__alpha_0.25",
                        "PPO_2024-04-20_16_34__alpha_0.5",
                        "PPO_2024-04-22_20_23__alpha_0.75",
                        "PPO_2024-04-22_08_37__alpha_1"]
SUMO_SEEDS = [39, 83, 49, 51, 74]

alpha_checkpoint_zipped = list(zip(ALPHAS_TO_TEST, CHECKPOINT_DIR_NAMES))

reward_folder_name = "combined_reward_with_delta_wait"

combined_reward_fn_factory = combined_reward_function_factory_with_delta_wait_time

# ---------------------- FTC TEST ------------------------------------------
for alpha, checkpoint_dir_name in alpha_checkpoint_zipped: 
    
    initial_path_to_store = f"reward_experiments/2x2grid/{reward_folder_name}/EVALUATION/{checkpoint_dir_name}/fixed_tc"
    
    combined_reward_fn = combined_reward_fn_factory(alpha)

    for sumo_seed in SUMO_SEEDS:
        
        path_to_store =  os.path.abspath(os.path.join(initial_path_to_store,f"SEED_{sumo_seed}_"))
        
        os.makedirs(path_to_store, exist_ok=True)

        csv_metrics_path = os.path.join(path_to_store, CSV_FILE_NAME)
        tb_log_dir = path_to_store

        # env_params is seed specific + has the fixed_ts param turned on 
        env_params_eval = init_env_params_2x2(reward_function=combined_reward_fn,
                                              observation_function=EntireObservationFunction,
                                              num_seconds=SIM_NUM_SECONDS,
                                              sumo_seed=sumo_seed, 
                                              render=True,
                                              fixed_ts=True)

        # ------------------ SAVE DATA AND STORE ------------------

        eval_config_info = {"checkpoint_dir_name_to_evaluate": checkpoint_dir_name,
                            "number_eval_steps": NUM_ENV_STEPS,
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
            par_pz_env_2x2_creator(env_params=env_params_eval,
                                   single_eval_mode=True,
                                   csv_path = csv_metrics_path,
                                   tb_log_dir=tb_log_dir))

        agent_ids = rllib_pz_env.get_agent_ids()

        # ----------- GET ADDITIONAL METRICS READY -----------
        
        extra_metrics_csv_path = os.path.join(path_to_store, EXTRA_METRICS_CSV_FILE_NAME) 
        extra_metrics_tb_log_dir = path_to_store

        with open(extra_metrics_csv_path,  "w", newline="") as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            headers = ["env_step_num"]
            agent_reward_headers = [f"reward_{agent_id}" for agent_id in agent_ids]
            total_reward_header = ["total_agent_reward"]
            
            headers += agent_reward_headers
            headers += total_reward_header

            headers = (headers)
            csv_writer.writerow(headers)

        tb_writer = SummaryWriter(extra_metrics_tb_log_dir)  # prep TensorBoard

        # ----------------- START TESTING ----------------------

        cum_reward = {agent_id : 0 for agent_id in agent_ids} 
        obs, info = rllib_pz_env.reset() 

        # this is dummy actions just to pass in so no exceptions are raised
        # different actions have NO effect, only used to pass the env wrapper checks and avoid exceptions being raised.
        # fixed tc is also the same across different reward functions - rewards do not get recorded
        actions = {'1':0, '2': 2, '5': 0, '6': 3}

        try:
            for i in range(NUM_ENV_STEPS):
                
                observations, rewards, terminations, truncations, infos = rllib_pz_env.step(actions)

                for agent_id in agent_ids:
                    cum_reward[agent_id] += rewards[agent_id]

        except Exception as e:
            # Handle any exceptions that might occur in the loop
            print(f"An exception occurred: {e}")
            raise

        else:      
            total_reward = sum(cum_reward.values())

            with open(extra_metrics_csv_path,  "a", newline="") as f:
                csv_writer = csv.writer(f, lineterminator='\n')
                data = ([NUM_ENV_STEPS] +
                        [cum_reward[agent_id] for agent_id in agent_ids] + 
                        [total_reward])
                csv_writer.writerow(data)
                
            print("Total reward: ", total_reward)

        finally:
            rllib_pz_env.close()