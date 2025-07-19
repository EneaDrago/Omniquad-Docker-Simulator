import yaml
import torch
import numpy as np
from rl_games.common.player_factory import PlayerFactory
import copy



def build_rlg_model(weights_path: str,
                       env_cfg_path: str,
                       agent_cfg_path: str,
                       device: str = "cuda:0") -> torch.nn.Module:
    """
    Carica un checkpoint rl‑games anche con la vecchia 1.6 (PlayerFactory).
    """
    # 1. carica YAML con Loader completo (serve per !!python/tuple)
    with open(agent_cfg_path, "r") as f:
        agent_cfg = yaml.load(f, Loader=yaml.Loader)
    with open(env_cfg_path, "r") as f:
        env_cfg = yaml.load(f, Loader=yaml.Loader)

    # 2. merge manuale
    merged_cfg = copy.deepcopy(agent_cfg)
    merged_cfg["params"]["env_config"] = env_cfg
    params = merged_cfg["params"]

    # 3. costruisci il player tramite factory
    algo_name = params["algo"]["name"]          # ← es. 'a2c_continuous'
    player = PlayerFactory().create(algo_name, params=params)
    player.restore(weights_path)                # carica pesi + normalizer

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