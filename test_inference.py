import torch
import numpy as np
from src.rlg_quad_controller.rlg_quad_controller.utils.rlg_utils import build_rlg_model, run_inference

# Carica modello
model = build_rlg_model(
    weights_path="src/rlg_quad_controller/models/omni_flat_v1/omniquad_flat.pth",
    env_cfg_path="src/rlg_quad_controller/models/omni_flat_v1/env.yaml",
    agent_cfg_path="src/rlg_quad_controller/models/omni_flat_v1/agent.yaml",
    device="cuda:0"
)

# Carica osservazione identica
obs = np.load("ossservazione_test.npy")
# Esegui inference
out = run_inference(model, obs, det=True)
print(out)
