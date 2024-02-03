import torch 

class Config:

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.state_dim = 84
        self.action_dim = 4
        
        # NN-specific info 
        self.lr = 1e-4
        self.no_hidden_layers = 100

        # Sampling 
        self.no_episodes = 5
        self.rollout_length = 200      # aka timesteps in each trajectory/rollout

        # Training 
        self.no_training_iterations = 3

        # optimisation
        self.optimisation_epochs = 10    # how many times same data should be used to reshuffle to minibatches and make optimisation updates 
        self.minibatch_size = 20        # minibatch_size < batch size * no_episodes 
        
        # PPO 
        self.discount = 0.97
        self.ppo_ratio_clip = 0.1
        self.gae_lamda = 0.95
        # self.max_steps = 4e5

        # logging
        self.logger_dir = 1
        self.play_only = False
        self.saved_checkpoint = 'checkpoint/models_info.txt'



        # self.log_interval = 100
        # self.gae_tau = 0.95
        # self.gradient_clip = 4.7
