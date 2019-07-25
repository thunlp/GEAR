import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from utils import load_bert_features_claim, correct_prediction
from models import GEAR

parser = argparse.ArgumentParser()
parser.add_argument('--patience', type=int, default=20, help='Patience')
parser.add_argument('--seed', type=int, default=314, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dir_path = '../outputs/gear-%devi-%dlayer-%s-%dseed-001' % (args.evi_num, args.layer, args.pool, args.seed)
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

if os.path.exists(dir_path + '/results.txt'):
    print(dir_path + ' results exists!')
    exit(0)
else:
    print(dir_path)

train_features, train_labels, train_claims = load_bert_features_claim('../data/gear/gear-train-set-0_001-features.tsv', args.evi_num)
dev_features, dev_labels, dev_claims = load_bert_features_claim('../data/gear/gear-dev-set-0_001-features.tsv', args.evi_num)

feature_num = train_features[0].shape[1]
model = GEAR(nfeat=feature_num, nins=args.evi_num, nclass=3, nlayer=args.layer, pool=args.pool)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    train_features.cuda()
    train_labels.cuda()
    dev_features.cuda()
    dev_labels.cuda()

train_features, train_labels, train_claims = Variable(train_features), Variable(train_labels), Variable(train_claims)
dev_features, dev_labels, dev_claims = Variable(dev_features), Variable(dev_labels), Variable(dev_claims)

train_data = TensorDataset(train_features, train_labels, train_claims)
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

dev_data = TensorDataset(dev_features, dev_labels, dev_claims)
dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size)

best_accuracy = 0.0
patience_counter = 0
best_epoch = 0

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    correct_pred = 0.0
    train_tqdm_iterator = tqdm(train_dataloader)
    for index, data in enumerate(train_tqdm_iterator):
        feature_batch, label_batch, claim_batch = data
        feature_batch = feature_batch.cuda()
        label_batch = label_batch.cuda()
        claim_batch = claim_batch.cuda()

        optimizer.zero_grad()
        outputs = model(feature_batch, claim_batch)

        loss = F.nll_loss(outputs, label_batch)

        correct_pred += correct_prediction(outputs, label_batch)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        description = 'Acc: %lf, Loss: %lf' % (correct_pred / (index + 1) / args.batch_size, running_loss / (index + 1))
        train_tqdm_iterator.set_description(description)

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct_pred / len(train_dataloader.dataset)
    print('Train total acc: %lf, total loss: %lf\r\n' % (train_accuracy, train_loss))

    model.eval()
    running_loss = 0.0
    correct_pred = 0.0
    dev_tqdm_iterator = tqdm(dev_dataloader)
    with torch.no_grad():
        for index, data in enumerate(dev_tqdm_iterator):
            feature_batch, label_batch, claim_batch = data
            feature_batch = feature_batch.cuda()
            label_batch = label_batch.cuda()
            claim_batch = claim_batch.cuda()

            outputs = model(feature_batch, claim_batch)
            loss = F.nll_loss(outputs, label_batch)

            correct_pred += correct_prediction(outputs, label_batch)
            running_loss += loss.item()

            description = 'Acc: %lf, Loss: %lf' % (correct_pred / (index + 1) / args.batch_size, running_loss / (index + 1))
            dev_tqdm_iterator.set_description(description)

    dev_loss = running_loss / len(dev_dataloader)
    dev_accuracy = correct_pred / len(dev_dataloader.dataset)
    print('Dev total acc: %lf, total loss: %lf\r\n' % (dev_accuracy, dev_loss))

    if dev_accuracy > best_accuracy:
        best_accuracy = dev_accuracy
        best_epoch = epoch
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'best_accuracy': best_accuracy,
                    'train_losses': train_loss,
                    'dev_losses': dev_loss},
                    '%s/best.pth.tar' % dir_path)
        patience_counter = 0
    else:
        patience_counter += 1

    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'best_accuracy': best_accuracy,
                'train_losses': train_loss,
                'dev_losses': dev_loss},
                '%s/epoch.%d.pth.tar' % (dir_path, epoch))

    if patience_counter > args.patience:
        print("Early stopping...")
        break

print(best_epoch)
print(best_accuracy)

fout = open(dir_path + '/results.txt', 'w')
fout.write('%d\t%lf\r\n' % (best_epoch, best_accuracy))
fout.close()