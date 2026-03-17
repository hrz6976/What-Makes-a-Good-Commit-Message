import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = REPO_ROOT / "Dataset" / "sampled messages.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


class ModelConfig:
    batch_size = 50
    output_size = 2
    hidden_dim = 384
    n_layers = 2
    lr = 1e-6
    bidirectional = False
    drop_prob = 0.55
    epochs = int(os.environ.get("CGOOD_EPOCHS", "10"))
    print_every = 10
    clip = 5
    use_cuda = USE_CUDA
    bert_path = "bert-base-uncased"
    sampleRate = float(os.environ.get("CGOOD_WHY_SAMPLE_RATE", "2"))
    whatSampleRate = float(os.environ.get("CGOOD_WHAT_SAMPLE_RATE", "1.5"))
    folds = int(os.environ.get("CGOOD_FOLDS", "10"))
    data_path = Path(os.environ.get("CGOOD_DATA_FILE", str(DEFAULT_DATA_FILE)))
    output_dir = Path(os.environ.get("CGOOD_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    save_path = output_dir / "cgood_tmp.pth"


class bert_lstm(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.bert = BertModel.from_pretrained(bertpath)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_size)

    def forward(self, x, hidden):
        attention_mask = (x != 0).long()
        x = self.bert(input_ids=x, attention_mask=attention_mask)[0]
        _, (hidden_last, _) = self.lstm(x, hidden)

        if self.bidirectional:
            hidden_last_out = torch.cat([hidden_last[-2], hidden_last[-1]], dim=-1)
        else:
            hidden_last_out = hidden_last[-1]
        out = self.dropout(hidden_last_out)
        return self.fc(out)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 2 if self.bidirectional else 1
        if USE_CUDA:
            hidden = (
                weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
            )
        else:
            hidden = (
                weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
            )
        return hidden


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction="sum"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, output, target):
        out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
        probs = torch.sigmoid(out_target)
        focal_weight = torch.pow(1 - probs, self.gamma)
        ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction="none")
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            focal_loss = (focal_loss / focal_weight.sum()).sum()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()
        return focal_loss


def train_model(config, data_train):
    net = bert_lstm(
        config.bert_path,
        config.hidden_dim,
        config.output_size,
        config.n_layers,
        config.bidirectional,
        config.drop_prob,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    if config.use_cuda:
        net.cuda()
    net.train()

    for e in range(config.epochs):
        h = net.init_hidden(config.batch_size)
        counter = 0
        for inputs, labels in data_train:
            counter += 1
            if config.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            h = tuple(each.data for each in h)
            net.zero_grad()
            output = net(inputs, h)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            optimizer.step()

            if counter % config.print_every == 0:
                print(
                    "Epoch: {}/{}, ".format(e + 1, config.epochs),
                    "Step: {}, ".format(counter),
                    "Loss: {:.6f}, ".format(loss.item()),
                )
    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), config.save_path)


def test_model(config, data_test):
    net = bert_lstm(
        config.bert_path,
        config.hidden_dim,
        config.output_size,
        config.n_layers,
        config.bidirectional,
        config.drop_prob,
    )
    state_dict = torch.load(config.save_path, map_location="cuda" if config.use_cuda else "cpu")
    net.load_state_dict(state_dict, strict=False)
    if config.use_cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    test_losses = []
    res = []
    h = net.init_hidden(config.batch_size)
    net.eval()

    for inputs, labels in data_test:
        h = tuple(each.data for each in h)
        if USE_CUDA:
            inputs, labels = inputs.cuda(), labels.cuda()
        output = net(inputs, h)
        test_loss = criterion(output.squeeze(), labels.long())
        test_losses.append(test_loss.item())
        _, pred = torch.max(output, 1)
        res.append(pred.cpu().numpy().tolist())

    res = np.array(res).reshape(1, -1).tolist()
    return res[0]


def myDataProcess(dataFile):
    df = pd.read_csv(dataFile, encoding="utf-8", encoding_errors="replace")
    multi_commit = df["if_mulit_commit"].astype("string").str.strip()
    labeledDF = df[df.label.notnull() & (multi_commit.isna() | (multi_commit == ""))].copy()
    labeledDF["new_message1"] = labeledDF["new_message1"].fillna("").apply(
        lambda x: x.replace("<enter>", "$enter")
        .replace("<tab>", "$tab")
        .replace("<url>", "$url")
        .replace("<version>", "$version")
        .replace("<pr_link>", "$pull request>")
        .replace("<issue_link >", "$issue")
        .replace("<otherCommit_link>", "$other commit")
        .replace("<method_name>", "$method")
        .replace("<file_name>", "$file")
        .replace("<iden>", "$token")
    )
    whyLabels = labeledDF["label"].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (1 if x == 2.0 else (0 if x == 3.0 else 1)))
    )
    whatLabels = labeledDF["label"].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (0 if x == 2.0 else (1 if x == 3.0 else 0)))
    )
    Labels = labeledDF["label"].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (0 if x == 2.0 else (0 if x == 3.0 else 1)))
    )
    print("load data successfully!")
    messages = list(labeledDF["new_message1"].array)
    return messages, np.array(whyLabels), np.array(whatLabels), np.array(Labels)


if __name__ == "__main__":
    model_config = ModelConfig()
    text, whyLabels, whatLabels, Labels = myDataProcess(model_config.data_path)
    tokenizer = BertTokenizer.from_pretrained(model_config.bert_path)
    result_comments_id = tokenizer(text, padding=True, truncation=True, max_length=200, return_tensors="pt")
    X = result_comments_id["input_ids"]
    y = torch.from_numpy(whyLabels).float()
    yWhat = torch.from_numpy(whatLabels).float()
    yAll = torch.from_numpy(Labels).float()

    fold = KFold(n_splits=model_config.folds, random_state=6666, shuffle=True)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    fold_num = 1

    for train_index, test_index in fold.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        print("train_label: %s" % str(sorted(Counter(y_train).items())))
        yWhat_train, yWhat_test = yWhat[train_index], yWhat[test_index]
        yAll_train, yAll_test = yAll[train_index], yAll[test_index]

        posNum = np.sum(whyLabels == 1)
        negNum = int(posNum / model_config.sampleRate)
        XWhy_train, y_train = RandomOverSampler(
            sampling_strategy={1: posNum, 0: negNum},
            random_state=666,
        ).fit_resample(X_train, y_train)
        print("Fold", fold_num, "why resampled_label:", str(sorted(Counter(y_train).items())))
        XWhy_train = torch.from_numpy(XWhy_train)
        y_train = torch.from_numpy(y_train)
        train_data = TensorDataset(XWhy_train, y_train)
        test_data = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=model_config.batch_size, drop_last=True)
        if USE_CUDA:
            print("Run on GPU.")
        else:
            print("No GPU available, run on CPU.")
        model_config.save_path = model_config.output_dir / f"cgood_fold{fold_num}_why.pth"
        train_model(model_config, train_loader)
        predWhy = test_model(model_config, test_loader)

        posNum = np.sum(whatLabels == 1)
        negNum = int(posNum / model_config.whatSampleRate)
        XWhat_train, yWhat_train = RandomOverSampler(
            sampling_strategy={1: posNum, 0: negNum},
            random_state=666,
        ).fit_resample(X_train, yWhat_train)
        print("Fold", fold_num, "what resampled_label:", str(sorted(Counter(yWhat_train).items())))
        XWhat_train = torch.from_numpy(XWhat_train)
        yWhat_train = torch.from_numpy(yWhat_train)
        train_data = TensorDataset(XWhat_train, yWhat_train)
        test_data = TensorDataset(X_test, yWhat_test)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=model_config.batch_size, drop_last=True)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=model_config.batch_size, drop_last=True)
        if USE_CUDA:
            print("Run on GPU.")
        else:
            print("No GPU available, run on CPU.")
        model_config.save_path = model_config.output_dir / f"cgood_fold{fold_num}_what.pth"
        train_model(model_config, train_loader)
        predWhat = test_model(model_config, test_loader)

        print(len(predWhat))
        print(len(predWhy))
        print(len(yAll_test))
        for i in range(0, len(predWhy)):
            why, what, tar = predWhy[i], predWhat[i], yAll_test[i]
            if (why + what) == 2 and tar == 1:
                tp += 1
            elif (why + what) == 2 and tar == 0:
                fp += 1
            elif (why + what) != 2 and tar == 1:
                fn += 1
            elif (why + what) != 2 and tar == 0:
                tn += 1

        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        negative_precision = 0 if tn + fn == 0 else tn / (tn + fn)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        negative_recall = 0 if tn + fp == 0 else tn / (tn + fp)
        f1 = 0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        negative_f1 = (
            0
            if (negative_precision + negative_recall) == 0
            else (2 * negative_recall * negative_precision) / (negative_precision + negative_recall)
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print(tp, fp, tn, fn)
        print("Total Accuracy", accuracy)
        print("Total Precision", [negative_precision, precision])
        print("Total Recall", [negative_recall, recall])
        print("Total F1", [negative_f1, f1])
        fold_num += 1
