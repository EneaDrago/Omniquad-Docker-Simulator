import yaml
import torch
import numpy as np
from rl_games.torch_runner import Runner
from rl_games.common import cfg_helper     # utility per unire/validare i dizionari



# TODO: device should be a parameter


def build_rlg_model(
        weights_path: str,
        env_cfg_path: str,
        agent_cfg_path: str,
        device: str = "cuda:0"
) -> torch.nn.Module:
    """
    Carica un checkpoint rl‑games (nuovo formato) e restituisce
    il modello PyTorch pronto per l'inferenza.

    Args:
        weights_path: percorso del file .pth salvato da rl‑games
        env_cfg_path: path a env.yaml (il “task”/MDP)
        agent_cfg_path: path a agent.yaml (algo/network/ppo ecc.)
        device:      'cuda:0', 'cpu', ecc.

    Returns:
        model (torch.nn.Module) in modalità eval() sul device indicato
    """
    # 1. leggo i due YAML
    with open(agent_cfg_path, "r") as f:
        agent_cfg = yaml.safe_load(f)
    with open(env_cfg_path, "r") as f:
        env_cfg = yaml.safe_load(f)

    # 2. rl‑games si aspetta un unico dict "params".
    #    Il runner farà poi   cfg['params']['env_config']   per passarlo al costruttore dell'ambiente
    #
    #    Qualche helper:
    #       - cfg_helper.ConfigMerger.merge ➜ preserva override / default
    #       - ma per casi semplici basta aggiungere la chiave a mano.
    #
    merged_cfg = agent_cfg.copy()            # ← contiene già la chiave top‑level "params"
    merged_cfg["params"]["env_config"] = env_cfg

    # 3. Costruiamo un runner "fittizio" solo per generare il Player
    runner = Runner(merged_cfg)
    player = runner.create_player()

    # 4. Carichiamo pesi e otteniamo il modello
    player.restore(weights_path)             # carica anche running‑mean, scaler, ecc.

    model = player.model.to(device)
    model.eval()
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