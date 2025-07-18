import yaml
import torch
import numpy as np
from rl_games.algos_torch.network_builder import A2CBuilder
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


# TODO: device should be a parameter

def build_rlg_model(weights_path, env_params, agent_params):

    weights = torch.load(weights_path, map_location=torch.device('cuda:0'),weights_only=False)

    net_params = agent_params['params']['network']

    network = A2CBuilder()
    network.load(net_params)

    model_a2c = ModelA2CContinuousLogStd(network)

    build_config = {
            'actions_num' : 12,     ###===============change manually===============###
            'input_shape' : (41,),  ###===============change manually===============###
            'num_seqs' : 1,
            'value_size': 1,
            'normalize_value' : agent_params['params']['config']['normalize_value'],
            'normalize_input': agent_params['params']['config']['normalize_input']
        }
    model = model_a2c.build(build_config)
    model.to('cuda:0')

    model.load_state_dict(weights['model'])

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