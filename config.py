import torch 

class Config:

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.state_dim = 84
        self.action_dim = 4
        
        # NN-specific info 
        self.no_hidden_layers = 100
        self.lr = 1e-4

        # Sampling 
        self.no_episodes = 3            # Next try 1400 episodes 
        self.rollout_length = 150       # Aka timesteps in each trajectory/rollout, next try 720 
                                        
        # Training 
        self.no_training_iterations = 100

        # Optimisation
        self.optimisation_epochs =  40   # how many times same data should be used to reshuffle to minibatches and make optimisation updates 
        self.minibatch_size = 100      # minibatch_size < batch size * no_episodes 
        
        # PPO 
        self.discount = 0.97
        self.ppo_ratio_clip = 0.1
        self.gae_lamda = 0.95

        # Logging
        # self.play_only = False
        # self.save_model_path = 