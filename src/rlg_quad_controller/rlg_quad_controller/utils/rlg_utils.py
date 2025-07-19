import yaml
import torch
import numpy as np
from rl_games.torch_runner import Runner
import copy



def build_rlg_model(weights_path: str,
                    env_cfg_path: str,
                    agent_cfg_path: str,
                    device: str = "cuda:0") -> torch.nn.Module:
    """
    Restituisce il modello PyTorch (eval) da un checkpoint rl‑games 1.6.x
    """
    # 1) leggi i due YAML con loader “completo” (serve per !!python/tuple)
    with open(agent_cfg_path) as f:
        agent_yaml = yaml.load(f, Loader=yaml.Loader)
    with open(env_cfg_path) as f:
        env_yaml   = yaml.load(f, Loader=yaml.Loader)

    # 2) mergia:   params['config']['env_config']  ← env_yaml
    params = copy.deepcopy(agent_yaml["params"])
    params["config"]["env_config"] = env_yaml

    # 3) Runner → load_config → create_player
    runner = Runner()                      # ctor senza argomenti
    runner.load_config(params=params)      # inizializza algo_name, ecc.
    player = runner.create_player()        # factory interna già registrata

    # 4) carica i pesi + normalizzatori
    player.restore(weights_path)

    # 5) prendi la rete e mettila sul device
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