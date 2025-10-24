from monai.networks.nets import SegResNet
import torch

# build the same network as in your bundle
net = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2
)

# load weights
state_dict = torch.load("model.pt", map_location="cpu")
net.load_state_dict(state_dict)
print("✅ Model loaded successfully")
