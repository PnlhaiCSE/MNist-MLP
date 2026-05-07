import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)

        return x


device = torch.device("cpu")

model = MLP().to(device)

model.load_state_dict(
    torch.load(
        "models/mnist_mlp_best.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully!")

dummy_input = torch.randn(1, 784)

torch.onnx.export(
    model,
    dummy_input,
    "models/mnist_mlp_best.onnx",

    export_params=True,
    opset_version=18,
    do_constant_folding=True,

    input_names=["input"],
    output_names=["output"],

    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("ONNX exported successfully!")