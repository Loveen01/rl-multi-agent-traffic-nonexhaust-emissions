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

from utils.environment_creator import par_pz_env_2x2_creator, init_env_params_2x2 
from utils.data_exporter import save_data_under_path

from torch.utils.tensorboard import SummaryWriter

from pprint import pprint

ray.shutdown()
ray.init()

# CONSTANTS
ENV_NAME = "2x2grid"
RLLIB_DEBUG_SEED = 10
NUM_EVAL_ITER = 3000
SIM_NUM_SECONDS = NUM_EVAL_ITER*5
CSV_FILE_NAME = "eval_metrics.csv" 

# manually collect checkpoint - CHANGE THESE VARS IF NECESSARY:
CHECKPOINT_NUM = '000099'

TRAINED_CHECKPOINT_PATH = os.path.abspath(
    f"reward_experiments/2x2grid/PPO_2024-04-19_19_25__alpha_0.25/PPO_2024-04-19_19_25__alpha_0.25/" +
    f"PPO_2x2grid_2ac30_00000_0_2024-04-19_19-25-15/checkpoint_{CHECKPOINT_NUM}"
)

CHECKPOINT_DIR_NAME = "PPO_2024-04-19_19_25__alpha_0.25"

ALPHA = 0.25

# main evaluation path
MANUAL_EVAL_PATH = os.path.abspath(f"reward_experiments/2x2grid/EVALUATION/{CHECKPOINT_DIR_NAME}")

SUMO_SEEDS = [39, 83, 49, 51, 74]

for SUMO_SEED in SUMO_SEEDS:
    # path to store trained
    path_to_store_trained_eval = os.path.join(MANUAL_EVAL_PATH, 
                                            "trained", 
                                            f"CHKPT_{int(CHECKPOINT_NUM)}_SEED_{SUMO_SEED}")
    
    csv_metrics_path_trained = os.path.abspath(os.path.join(path_to_store_trained_eval, CSV_FILE_NAME))    
    tb_log_dir_trained = os.path.abspath(path_to_store_trained_eval)

    os.makedirs(path_to_store_trained_eval, exist_ok=True)

    # initialise reward fn
    combined_reward_fn = combined_reward_function_factory(ALPHA)

    # configure env params
    env_params_eval = init_env_params_2x2(reward_function=combined_reward_fn,
                                        observation_function=EntireObservationFunction, 
                                        num_seconds=SIM_NUM_SECONDS, 
                                        seed=SUMO_SEED, 
                                        render_mode="human")

    # save info and store
    config_info = {"checkpoint_path_to_evaluate": TRAINED_CHECKPOINT_PATH,
                   "sumo_eval_seed": SUMO_SEED,
                   "number_eval_iterations": NUM_EVAL_ITER,
                   "congestion_coeff": ALPHA,
                   "evaluation_environment_args" : env_params_eval}

    try:
        save_data_under_path(config_info,
                            path_to_store_trained_eval,
                            "evaluation_info.json")
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving evaluation info: {e}") from e

    # just to get possible agents, no use elsewhere
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
    
    CSV_FILE_NAME_ADDIT = "additional_metrics.csv"
    csv_metrics_path_trained_additional_metrics = os.path.abspath(os.path.join(path_to_store_trained_eval, CSV_FILE_NAME_ADDIT))    
    tb_log_dir_trained_additional_metrics = os.path.abspath(path_to_store_trained_eval)

    with open(csv_metrics_path_trained_additional_metrics,  "w", newline="") as f:
        csv_writer = csv.writer(f, lineterminator='\n')
        headers = ["env_step_num"]
        reward_headers = [f"reward_{agent_id}" for agent_id in par_env_no_agents]
        headers += reward_headers
        headers = (headers)
        csv_writer.writerow(headers)

    tb_writer = SummaryWriter(tb_log_dir_trained_additional_metrics)  # prep TensorBoard

    # ----------- TESTING TRAINED CHECKPOINT -----------

    trained_algo = config.build().from_checkpoint(TRAINED_CHECKPOINT_PATH, 
                                    policy_ids=par_env_no_agents, 
                                    policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))

    par_env = ParallelPettingZooEnv(
        par_pz_env_2x2_creator(env_params=env_params_eval, 
                            single_eval_mode=True,
                            csv_path = csv_metrics_path_trained,
                            tb_log_dir=tb_log_dir_trained))

    cum_reward = {agent_id:0 for agent_id in par_env_no_agents} 
    reward = {agent_id:0 for agent_id in par_env_no_agents} 

    obs, info = par_env.reset()

    for eval_i in range(NUM_EVAL_ITER):

        actions = {}

        for agent_id in par_env_no_agents:
            action, state_outs, infos = trained_algo.get_policy(agent_id).compute_actions(obs[agent_id].reshape((1,84)))
            actions[agent_id] = action.item()

        obs, rews, terminateds, truncateds, infos = par_env.step(actions)
        
        for agent_id in par_env_no_agents:
            cum_reward[agent_id] += rews[agent_id]
            reward[agent_id] = rews[agent_id]
        
        if eval_i == NUM_EVAL_ITER-1:
            with open(csv_metrics_path_trained_additional_metrics,  "a", newline="") as f:
                csv_writer = csv.writer(f, lineterminator='\n')
                data = ([eval_i] +
                        [cum_reward[agent_id] for agent_id in par_env_no_agents])
                csv_writer.writerow(data)

    par_env.close()

    total_reward = sum(cum_reward.values())

    pprint("Agent reward: ", reward)
    pprint("Total reward: ", total_reward)
