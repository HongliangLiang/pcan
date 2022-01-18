# coding=utf-8
import warnings

import pandas
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from data import TreeDataset
from models.avg_max import CodeCloneClassifier
from pipeline import Pipeline
from utils import get_params, collate_fn

# from torch.nn.TransformerEncoder

warnings.filterwarnings('ignore')


def validate(model, val_loader, loss_func):
    model.eval()

    predicts = []
    trues = []
    total_loss = 0.0
    with torch.no_grad():
        for step, (x, y) in enumerate(val_loader):
            y = y.to(device)
            y_pred = model(x)

            val_loss = loss_func(y_pred, y)

            predicts.extend((y_pred.data > 0.5).cpu().detach().numpy())
            trues.extend(y.cpu().detach().numpy())

            total_loss += val_loss.item() * len(y)
    p, r, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    with open('./outputs/eval_result.txt', 'at') as file:
        predicts = np.array(predicts)
        trues = np.array(trues)
        false: pandas.DataFrame = val_loader.dataset.pairs.loc[predicts != trues]
        s = false.to_string(header=False, index=False)
        file.write(s)
        file.write("\n")

    return total_loss, p, r, f1


def train(model, train_loader, optimizer, loss_func):
    model.train()
    training_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y = y.to(device)
        y_pred = model(x)

        loss = loss_func(y_pred, y)

        loss.backward()
        optimizer.step()
        training_loss += loss.item() * len(y)
        if step % 100 == 0:
            print(f'step: {step} || loss: {loss} || total train loss: {training_loss}')
    return training_loss


def evaluate(args):
    p = Pipeline(args)
    vocab = p.get_torchtext_vocab()
    loss_func = torch.nn.BCELoss()
    state_dict = torch.load(args['model_path'], map_location='cpu')
    model = CodeCloneClassifier(args, vocab).to(device)
    model.load_state_dict(state_dict)
    val_loader = DataLoader(TreeDataset(mode='eval', args=args, ), collate_fn=collate_fn, batch_size=args['batch_size'], shuffle=True, num_workers=5, pin_memory=True)
    val_loss, p, r, f1 = validate(model, val_loader, loss_func)
    print(f"\tval:{val_loss:.2f},"
              f"\tp:{p:.4f}"
              f"\tr:{r:.4f}"
              f"\tf1:{f1:.4f}")


def main(args):
    p = Pipeline(args)
    vocab = p.get_torchtext_vocab()

    train_loader = DataLoader(TreeDataset(mode='train', args=args), collate_fn=collate_fn,
                              batch_size=args['batch_size'], shuffle=True, num_workers=5,
                              pin_memory=True, drop_last=True)

    val_loader = None

    loss_func = torch.nn.BCELoss()

    model = CodeCloneClassifier(args, vocab).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True, factor=0.5, min_lr=1e-7)

    best_val_loss = 1000000
    val_loss, p, r, f1 = 0, 0, 0, 0
    for i in range(args['epochs']):
        print(f'epoch: {i}')

        training_loss = train(model, train_loader, optimizer, loss_func)

        if val_loader is None:
            val_loader = DataLoader(TreeDataset(mode='eval', args=args, ), collate_fn=collate_fn,
                                    batch_size=64,
                                    shuffle=False,
                                    num_workers=5, pin_memory=True)

        val_loss, p, r, f1 = validate(model, val_loader, loss_func)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'outputs/models/pcan_{i}_{best_val_loss}_{f1}.pt')
        scheduler.step(val_loss)

        print(f"train:{training_loss:.2f}"
              f"\tval:{val_loss:.2f},"
              f"\tp:{p:.4f}"
              f"\tr:{r:.4f}"
              f"\tf1:{f1:.4f}")


if __name__ == '__main__':
    import dgl
    import random

    seed = 26
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    params = vars(get_params())

    if not torch.cuda.is_available():
        setattr(params, 'device', 'cpu')
    params['n_etypes'] = 4

    device = torch.device(params['device'])
    print(params)
    if params['mode'] == 'train':
        main(params)
    elif params['mode'] == 'eval':
        evaluate(params)
    else:
        raise
