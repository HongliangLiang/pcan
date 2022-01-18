import argparse
import collections

import dgl
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

container_abcs = collections.abc


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None

        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)

    elif isinstance(elem, dgl.DGLGraph):
        return dgl.batch(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size", type=int, default=128, metavar='N',
                        help='hidden layer size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=7000)
    parser.add_argument('--dataset_name', type=str,
                        choices=['bcb', 'gcj', 'ojclone', 'tbccd', 'tbccd_2', 'tbccd_3', 'fa-ast', 'poj_104'],
                        default='fa-ast')
    parser.add_argument('--embedding_size', type=int, default=128)

    parser.add_argument('--type', type=int, default=5,
                        help='clone type, this arg is only valid for bcb')
    parser.add_argument('--task', type=str, choices=['clone', 'classification'], default='clone')

    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=6)
    parser.add_argument('--balanced', action='store_true', default=False)
    parser.add_argument('--no_data', action='store_true', default=False)
    parser.add_argument('--no_child', action='store_true', default=False)
    parser.add_argument('--ggnn', action='store_true', default=True)
    parser.add_argument('--gap', action='store_true', default=False)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--out_feats', type=int, default=128)
    parser.add_argument('--in_feats', type=int, default=128)
    parser.add_argument('--ms', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--path', type=str)
    parser.add_argument('--model_path', type=str, default='outputs/models/model.pt')
    parser.add_argument('--pure-graph', action='store_true', default=False)
    parser.add_argument('--pure-seq', action='store_true', default=False)

    args, _ = parser.parse_known_args()
    return args


def validate(model: torch.nn.Module, val_loader, loss_func, device):
    model.eval()

    predicts = []
    trues = []
    total_loss = 0.0
    with torch.no_grad():
        for (x, y) in val_loader:
            y = y.to(device)
            y_pred = model(x)

            val_loss = loss_func(y_pred, y)

            predicts.extend((y_pred.data > 0.5).cpu().detach().numpy())
            trues.extend(y.cpu().detach().numpy())

            total_loss += val_loss.item() * len(y)

    p, r, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    return total_loss, p, r, f1


def save_model(model, total_loss, best_val_loss, f1):
    if total_loss < best_val_loss:
        torch.save(model.state_dict(), f'/mnt/f1_{f1:4.f}_loss_{total_loss:4.f}.pt')
        print(f'save model to /mnt/f1_{f1:4.f}_loss_{total_loss:4.f}.pt')
