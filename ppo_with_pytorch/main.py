import torch 
import os 

from sumo_rl.environment.env import env, parallel_env, SumoEnvironment
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility

from environment.envs import RealMultiAgentSumoEnv
from environment.observation import Grid2x2ObservationFunction, EntireObservationFunction
from environment.reward_functions import combined_reward

from config import Config
from ppo_trainer import PPO_Trainer
from buffer import Buffer, EpisodeBuffer
from utils import utils

os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


env_folder = "data/2x2grid"

multi_agent_env = parallel_env(    
        net_file = os.path.join(env_folder, "2x2.net.xml"),
        route_file = os.path.join(env_folder, "2x2.rou.xml"),
        reward_fn = combined_reward,
        observation_class = EntireObservationFunction, 
        out_csv_name="outputs/2x2grid/ppo", 
        num_seconds=100000000000000000000,
        add_per_agent_info=True,
        add_system_info=True,
        single_agent=False)

configs = Config()

current_time = datetime.now().strftime("%b%d_%H-%M-%S")

log_dir = current_time

experiment_path = f"runs/{log_dir}"
logger = SummaryWriter(f"runs/{log_dir}") # will change every time you run this class 

config_txt_file_path = os.path.join(experiment_path, "config.txt")
utils.save_configs(configs, config_txt_file_path)

buffer = Buffer(multi_agent_env.max_num_agents, configs.rollout_length, configs.state_dim, configs.discount, configs.gae_lamda)
episode_buffer = EpisodeBuffer(multi_agent_env.max_num_agents, configs.no_episodes, configs.rollout_length, configs.state_dim)

ppo_trainer = PPO_Trainer(multi_agent_env, configs, buffer, episode_buffer, logger, experiment_path)

ppo_trainer.run()    