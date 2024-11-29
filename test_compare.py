import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model import Transformer
from FInal.config_reader import Config


def init_weights(m, seed=42):
    if seed is not None:
        torch.manual_seed(seed)

    shape_dict = {}

    for name, param in m.named_parameters():
        shape = tuple(param.shape)
        if shape not in shape_dict:
            if "weight" in name and param.dim() >= 2:
                # print("xavier normal weights  ",name)
                nn.init.xavier_normal_(param)  # Using Xavier normal initialization
                param.data = param.data.type(torch.float16)
            else:
                nn.init.constant_(param, 1.0)
                param.data = param.data.type(torch.float16)
            shape_dict[shape] = param.clone()
        else:
            param.data = shape_dict[shape].clone()


# def init_weights(m, seed=42):
#     if seed is not None:
#         torch.manual_seed(seed)

#     shape_dict = {}

#     for name, param in m.named_parameters():
#         shape = tuple(param.shape)
#         if shape not in shape_dict:
#             if 'weight' in name:
#                 nn.init.normal_(param, mean=3.0, std=0.1)
#                 param.data = param.data.type(torch.bfloat16)
#             elif 'bias' in name:
#                 nn.init.constant_(param, 1.0)
#                 param.data = param.data.type(torch.bfloat16)
#             shape_dict[shape] = param.clone()
#         else:
#             param.data = shape_dict[shape].clone()


if __name__ == "__main__":
    config = Config()
    vocabSize = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = Transformer(config.train, vocabSize, dtype=torch.float16).to(device)
    transformer.apply(init_weights)

    for name, params in transformer.named_parameters():
        print(f"Parameter: {name}, Shape: {params.shape}, Mean: {params.mean().item()}")

    # x_train = [torch.tensor([1, 2, 3, 2, 3])]

    # y_train = [torch.tensor(1)]

    x_train = [
        torch.tensor([1, 2, 3, 2, 3]),
        torch.tensor([3, 2, 1, 3, 2]),
        torch.tensor([2, 3, 1, 2, 3]),
    ]
    y_train = [torch.tensor(1), torch.tensor(1), torch.tensor(1)]

    dataset = TensorDataset(torch.stack(x_train), torch.stack(y_train))

    dataloader = DataLoader(
        dataset, batch_size=config.train["batch_size"], num_workers=1
    )

    x, _ = next(iter(dataloader))
    x = x.to(device)
    output = transformer.forward(x)
    print(output.shape)
    print(output)
