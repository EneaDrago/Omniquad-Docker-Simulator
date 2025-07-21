import yaml
import numpy as np
from rl_games.torch_runner import Runner
import copy
from rl_games.common import env_configurations
from types import SimpleNamespace
from gym.spaces import Box

import torch
import numpy.core.multiarray
from torch.serialization import add_safe_globals
from rl_games.algos_torch.torch_ext import safe_load



def build_rlg_model(weights_path: str,
                    env_cfg_path: str,
                    agent_cfg_path: str,
                    device: str = "cuda:0") -> torch.nn.Module:


    # --- patch temporanea torch.load con safe globals ---
    add_safe_globals([numpy.core.multiarray.scalar])
    original_torch_load = torch.load

    def patched_torch_load(f, *args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(f, *args, **kwargs)

    torch.load = patched_torch_load

    try:
        with open(agent_cfg_path) as f:
            agent_yaml = yaml.load(f, Loader=yaml.Loader)
        with open(env_cfg_path) as f:
            env_yaml = yaml.load(f, Loader=yaml.Loader)

        params = copy.deepcopy(agent_yaml["params"])
        params["config"]["env_config"] = env_yaml

        obs_dim = 41
        act_dim = 12

        if 'rlgpu' not in env_configurations.configurations:
            dummy_env = SimpleNamespace()
            dummy_env.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            dummy_env.action_space = Box(
                low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
            )
            env_configurations.configurations['rlgpu'] = {
                'env_creator': lambda **kwargs: dummy_env
            }

        runner = Runner()
        runner.load_config(params=params)
        player = runner.create_player()

        player.restore(weights_path)

        model = player.model.to(device).eval()
        return model
    finally:
        # Restore torch.load after model is loaded
        torch.load = original_torch_load




def run_inference(model, observation, det=True):
    with torch.no_grad():
        obs_tensor = torch.from_numpy(observation).to('cuda:0').float()
        obs_dict = {
            'is_train':    False,
            'prev_actions': None,
            'obs':         obs_tensor,
            'rnn_states':  None
        }
        action_dict = model(obs_dict)
        actions = action_dict['mus'] if det else action_dict['actions']
        return actions.cpu().numpy()


def run_inference_dict(model, observation):
    with torch.no_grad():
        obs_tensor = torch.from_numpy(observation).to('cuda:0').float()
        return model({
            'is_train':    False,
            'prev_actions': None,
            'obs':         obs_tensor,
            'rnn_states':  None
        })
