import yaml
import torch
import numpy as np
from rl_games.torch_runner import Runner
import copy
from rl_games.common import env_configurations



def build_rlg_model(weights_path: str,
                    env_cfg_path: str,
                    agent_cfg_path: str,
                    device: str = "cuda:0") -> torch.nn.Module:
    
    with open(agent_cfg_path) as f:
        agent_yaml = yaml.load(f, Loader=yaml.Loader)
    with open(env_cfg_path) as f:
        env_yaml   = yaml.load(f, Loader=yaml.Loader)

    params = copy.deepcopy(agent_yaml["params"])
    params["config"]["env_config"] = env_yaml

    # REGISTRA 'rlgpu' dummy env se non esiste
    if 'rlgpu' not in env_configurations.configurations:
        env_configurations.configurations['rlgpu'] = {
            'env_creator': lambda **kwargs: None
        }

    runner = Runner()
    runner.load_config(params=params)
    player = runner.create_player()
    player.restore(weights_path)

    model = player.model.to(device).eval()
    return model



def run_inference(model, observation, det=True):
    """
    Runs inference on a model given an observation.

    Args:
        model: A PyTorch model.
        observation: A numpy array containing the observation.

    Returns:
        A numpy array containing the action.
    """
    
    with torch.no_grad():
        obs_tensor = torch.from_numpy(observation).to('cuda:0').type(torch.float32)
        obs_dict = {'is_train': False,
                    'prev_actions': None,
                    'obs': obs_tensor,
                    'rnn_states': None}
        action_dict = model(obs_dict)
        actions = action_dict['mus'] if det else action_dict['actions']
        actions = actions.cpu().numpy()

    return actions



def run_inference_dict(model, observation):
    """
    Runs inference on a model given an observation.

    Args:
        model: A PyTorch model.
        observation: A numpy array containing the observation.

    Returns:
        The action dictionary.
    """
    
    with torch.no_grad():
        obs_tensor = torch.from_numpy(observation).to('cuda:0').type(torch.float32)
        obs_dict = {'is_train': False,
                    'prev_actions': None,
                    'obs': obs_tensor,
                    'rnn_states': None}
        action_dict = model(obs_dict)
        
    return action_dict