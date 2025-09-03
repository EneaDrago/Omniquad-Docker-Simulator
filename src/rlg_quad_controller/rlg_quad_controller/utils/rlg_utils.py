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



def build_rlg_player(weights_path: str,
                    env_cfg_path: str,
                    agent_cfg_path: str,
                    device: str = "cuda:0"):


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

        player.model.to(device).eval()
        return player
    finally:
        # Restore torch.load after model is loaded
        torch.load = original_torch_load




def run_inference(player, observation, det=True):
    obs_torch = torch.from_numpy(observation).to('cuda:0').float()
    action = player.get_action(obs_torch, is_deterministic=det)
    return action.cpu().numpy()


def run_inference_dict(model, observation):
    with torch.no_grad():
        obs_tensor = torch.from_numpy(observation).to('cuda:0').float()
        return model({
            'is_train':    False,
            'prev_actions': None,
            'obs':         obs_tensor,
            'rnn_states':  None
        })



def run_inference_chat(player, observation, det: bool = True):
    """
    Esegue l'inferenza in modo robusto.
    - det=True: usa la mean della distribuzione (niente sampling ⇒ niente std negativa).
    - det=False: sampling stocastico, con clamp minimo sulla std.
    Accetta observation con shape (obs_dim,) o (1, obs_dim); ritorna (act_dim,).
    """
    # Sanifica e sistemazione shape
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, ...]  # (1, obs_dim)

    # Evita NaN/Inf che fanno esplodere la sigma
    if not np.isfinite(obs).all():
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    obs_t = torch.from_numpy(obs).to(player.device)

    with torch.no_grad():
        out = player.model({'obs': obs_t})

        # rl-games di solito restituisce un dict con la distribuzione
        if isinstance(out, dict):
            if 'action_distribution' in out:
                distr = out['action_distribution']
                # Clamp di sicurezza sulla std (se presente come .scale)
                if hasattr(distr, 'scale'):
                    distr.scale.clamp_(min=1e-6)
                action_t = distr.mean if det else distr.sample()
            elif 'action' in out:  # fallback: alcuni modelli danno già l'azione
                action_t = out['action']
            else:
                # ultimo fallback: se il dict ha altro formato
                raise RuntimeError("Output del model non riconosciuto: manca 'action_distribution'/'action'")
        else:
            # Se il modello restituisce direttamente un tensore azione
            if torch.is_tensor(out):
                action_t = out
            else:
                raise RuntimeError("Output del model non riconosciuto")

    # shape finale (act_dim,)
    return action_t.squeeze(0).detach().cpu().numpy()
