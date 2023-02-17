import torch
import paddle

params = torch.load('RWKV-4-Pile-1B5-EngChn-testNovel-301-ctx2048-20230214.pth', map_location='cpu')

state_dict = {}
for k, v in params.items():
    state_dict[k] = v.float().numpy()

paddle.save(state_dict, 'RWKV-4-Pile-1B5-EngChn-testNovel-301-ctx2048-20230214.pdparams')