from utils.environment_creator import par_env_2x2_creator


env = par_env_2x2_creator()


# run_config=air.RunConfig(
#         stop=stop,
#         checkpoint_ppo_config=air.CheckpointConfig(
#         checkpoint_frequency=10,
#         ),
#         storage_path="/Users/loveen/Desktop/Masters project/rl-multi-agent-traffic-nonexhaust-emissions/results"
# )
# tune.Tuner("PPO", run_config=run_config, param_space=ppo_config).fit()
# ppo_algo = ppo_config.build()

# for i in range(1):
#         ppo_algo.train()

# ppo_algo.save_checkpoint(save_directory)