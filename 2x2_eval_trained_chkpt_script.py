import os 
import csv
import pandas as pd
import numpy as np
import random 
import re 

import ray 
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms import ppo 
from ray.rllib.algorithms.algorithm import Algorithm

from environment.reward_functions import combined_reward_function_factory 
from environment.observation import EntireObservationFunction

from utils.environment_creator import par_pz_env_2x2_creator, init_env_params_2x2 
from utils.data_exporter import save_data_under_path
from utils.utils import extract_folder_from_path

from torch.utils.tensorboard import SummaryWriter

from pprint import pprint


ray.shutdown()
ray.init()

# -------------------- CONSTANTS -------------------------
ENV_NAME = "2x2grid"
RLLIB_DEBUG_SEED = 10
NUM_EVAL_ITER = 1000
SIM_NUM_SECONDS = NUM_EVAL_ITER*5
sumo_seeds = [39, 83, 49, 51, 74]
CSV_FILE_NAME = "eval_metrics.csv" 

# -------------------- CHANGE THESE VARS ----------------------
TRAINED_CHECKPOINT_PATHS = ["reward_experiments/2x2grid/TRAINING/PPO_2024-04-18_12_59__alpha_0/PPO_2024-04-18_12_59__alpha_0/" + 
                            "PPO_2x2grid_24972_00000_0_2024-04-18_12-59-43/checkpoint_000099",

                            "reward_experiments/2x2grid/TRAINING/PPO_2024-04-19_19_25__alpha_0.25/PPO_2024-04-19_19_25__alpha_0.25/" +
                            "PPO_2x2grid_2ac30_00000_0_2024-04-19_19-25-15/checkpoint_000099",

                            "reward_experiments/2x2grid/TRAINING/PPO_2024-04-20_16_34__alpha_0.5/PPO_2024-04-20_16_34__alpha_0.5/" +
                            "PPO_2x2grid_85399_00000_0_2024-04-20_16-34-47/checkpoint_000099",
    
                            "reward_experiments/2x2grid/TRAINING/PPO_2024-04-22_20_23__alpha_0.75/PPO_2024-04-22_20_23__alpha_0.75/" + 
                            "PPO_2x2grid_bd816_00000_0_2024-04-22_20-23-03/checkpoint_000099",

                            "reward_experiments/2x2grid/TRAINING/PPO_2024-04-22_08_37__alpha_1/PPO_2024-04-22_08_37__alpha_1/" +
                            "PPO_2x2grid_22c3b_00000_0_2024-04-22_08-37-13/checkpoint_000099"]

TRAINED_CHECKPOINT_PATHS_ABS = [os.path.abspath(x) for x in TRAINED_CHECKPOINT_PATHS]
                        
# pattern = re.compile(r"PPO_2024-\d{2}-\d{2}_\d{2}_\d{2}__alpha_0?(\.\d+)?")

alphas_to_test = [0, 0.25, 0.5, 0.75, 1]
alpha_checkpoint_zipped = list(zip(alphas_to_test, TRAINED_CHECKPOINT_PATHS_ABS))
CHECKPOINT_DIR_NAME_ON_PATH_INDEX = 3

# ----------- CHANGE THESE VARS (FOR PARTIAL TESTING) -------------
PARTIAL_SUMO_SEEDS = [39]
    
for alpha, checkpoint_path in alpha_checkpoint_zipped:
    
    checkpoint_dir_name = extract_folder_from_path(checkpoint_path, CHECKPOINT_DIR_NAME_ON_PATH_INDEX)
    
    path_to_store = os.path.abspath(
        f"reward_experiments/2x2grid/EVALUATION/{checkpoint_dir_name}/trained")
    
    checkpoint_num = checkpoint_path[-2:]
    
    # initialise reward fn
    combined_reward_fn = combined_reward_function_factory(alpha)

    for sumo_seed in PARTIAL_SUMO_SEEDS:
        
        # path to store trained
        path_to_store_specific_seed_abs = os.path.abspath(os.path.join(path_to_store,
                                                                   f"CHKPT_{int(checkpoint_num)}_SEED_{sumo_seed}"))
        
        os.makedirs(path_to_store_specific_seed_abs, exist_ok=True)

        csv_metrics_path = os.path.join(path_to_store_specific_seed_abs, CSV_FILE_NAME)
        tb_log_dir = path_to_store_specific_seed_abs

        # configure env params
        env_params_eval = init_env_params_2x2(reward_function=combined_reward_fn,
                                              observation_function=EntireObservationFunction,
                                              num_seconds=SIM_NUM_SECONDS,
                                              sumo_seed=sumo_seed,
                                              render=True)
       
        # ------------------ SAVE DATA AND STORE ------------------

        config_info = {"checkpoint_path_to_evaluate": checkpoint_path,
                       "number_eval_iterations": NUM_EVAL_ITER,
                       "congestion_coeff": alpha,
                       "evaluation_environment_args" : env_params_eval}

        try:
            save_data_under_path(config_info,
                                path_to_store_specific_seed_abs,
                                "evaluation_info.json")
        except Exception as e:
            raise RuntimeError(f"An error occurred while saving evaluation info: {e}") from e

        # ----------- REGISTER ENV ----------------
        par_env_no_agents = par_pz_env_2x2_creator(env_params=env_params_eval).possible_agents

        register_env(ENV_NAME, lambda config: ParallelPettingZooEnv(
            par_pz_env_2x2_creator(env_params=env_params_eval)))

        config: PPOConfig
        # From https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/atari-ppo.yaml
        config: PPOConfig
        config = (
            PPOConfig()
            .environment(
                env=ENV_NAME,
                env_config=env_params_eval)
            .rollouts(
                num_rollout_workers = 1 # for sampling only 
            )
            .framework(framework="torch")
            .training(
                lambda_=0.95,
                kl_coeff=0.5,
                clip_param=0.1,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                train_batch_size=1000, # 1 env step = 5 num_seconds
                sgd_minibatch_size=500,
            )
            .debugging(seed=RLLIB_DEBUG_SEED) # identically configured trials will have identical results.
            .reporting(keep_per_episode_custom_metrics=True)
            .multi_agent(
                policies=set(par_env_no_agents),
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
                count_steps_by = "env_steps"
            )
            .fault_tolerance(recreate_failed_workers=True)
        )

        # ----------- GET ADDITIONAL METRICS READY -----------
        
        extra_metrics_csv_file_name = "extra_metrics.csv"
        extra_metrics_csv_path = os.path.join(path_to_store_specific_seed_abs, extra_metrics_csv_file_name)  
        extra_metrics_tb_log_dir = path_to_store_specific_seed_abs

        with open(extra_metrics_csv_path,  "w", newline="") as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            headers = ["env_step_num"]
            agent_reward_headers = [f"reward_{agent_id}" for agent_id in par_env_no_agents]
            total_reward_header = ["total_agent_reward"]
            
            headers += agent_reward_headers
            headers += total_reward_header

            headers = (headers)
            csv_writer.writerow(headers)

        tb_writer = SummaryWriter(extra_metrics_tb_log_dir)  # prep TensorBoard

        # ----------- TESTING TRAINED CHECKPOINT -----------

        checkpoint_path_abs = os.path.abspath(checkpoint_path)

        trained_algo = config.build().from_checkpoint(checkpoint_path_abs,
                                                      policy_ids=par_env_no_agents,
                                                      policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))

        par_env = ParallelPettingZooEnv(
            par_pz_env_2x2_creator(env_params=env_params_eval,
                                   single_eval_mode=True,
                                   csv_path = csv_metrics_path,
                                   tb_log_dir=tb_log_dir))

        cum_reward = {agent_id:0 for agent_id in par_env_no_agents} 

        obs, info = par_env.reset()

        try:
            for eval_i in range(NUM_EVAL_ITER):

                actions = {}

                for agent_id in par_env_no_agents:
                    action, state_outs, infos = trained_algo.get_policy(agent_id).compute_actions(obs[agent_id].reshape((1,84)))
                    actions[agent_id] = action.item()

                obs, rews, terminateds, truncateds, infos = par_env.step(actions)
                
                for agent_id in par_env_no_agents:
                    cum_reward[agent_id] += rews[agent_id]
        
        except Exception as e:
            # Handle any exceptions that might occur in the loop
            print(f"An exception occurred: {e}")
            raise

        else:      
            total_reward = sum(cum_reward.values())

            with open(extra_metrics_csv_path,  "a", newline="") as f:
                csv_writer = csv.writer(f, lineterminator='\n')
                data = ([NUM_EVAL_ITER] +
                        [cum_reward[agent_id] for agent_id in par_env_no_agents] + 
                        [total_reward])
                csv_writer.writerow(data)
                
            print("Total reward: ", total_reward)

        finally:
            par_env.close()