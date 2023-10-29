import os
import sys
import timeit
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, mean_squared_error

import wandb

import preprocess as pp


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.N_fingerprints = N_fingerprints
        self.dim = dim
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        if task == 'classification':
            self.W_property = nn.Linear(dim, 2)
        if task == 'regression':
            self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

    def mlp(self, vectors):
        """Classifier or regressor based on multilayer perceptron."""
        for l in range(self.layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs

    def forward_classifier(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_labels = torch.cat(data_batch[-1])

        if train:
            molecular_vectors = self.gnn(inputs)
            predicted_scores = self.mlp(molecular_vectors)
            loss = F.cross_entropy(predicted_scores, correct_labels)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_scores = self.mlp(molecular_vectors)
            predicted_scores = predicted_scores.to('cpu').data.numpy()
            predicted_scores = [s[1] for s in predicted_scores]
            correct_labels = correct_labels.to('cpu').data.numpy()
            return predicted_scores, correct_labels

    def forward_regressor(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return predicted_values, correct_values

    def predict(self, data_batch):

        inputs = data_batch
        self.eval()
        with torch.no_grad():
            molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
        predicted_values = predicted_values.to('cpu').data.numpy()
        predicted_values = predicted_values.reshape(-1)
        return list(predicted_values)


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            if task == 'classification':
                loss = self.model.forward_classifier(data_batch, train=True)
            if task == 'regression':
                loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_classifier(self, dataset):
        N = len(dataset)
        P, C = [], []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_scores, correct_labels = self.model.forward_classifier(
                                               data_batch, train=False)
            P.append(predicted_scores)
            C.append(correct_labels)
        AUC = roc_auc_score(np.concatenate(C), np.concatenate(P))
        return AUC

    def test_regressor(self, dataset):
        N = len(dataset)
        rmse_list = []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_values, correct_values = self.model.forward_regressor(
                                               data_batch, train=False)
            rmse_list.append(mean_squared_error(correct_values, predicted_values, squared=False))
        return np.mean(rmse_list)

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

def predict(model, dataset):
    submit_path = "../data/sample_submission.csv" # change your path (relative path on teminal location)
    submit = pd.read_csv(submit_path)
    pred_lst = []
    N = len(dataset)
    for i in range(0, N, batch_test):
        data_batch = list(zip(*dataset[i:i+batch_test]))
        pred = model.predict(data_batch)
        pred_lst += pred
    submit["MLM"] = pred_lst
    submit.to_csv("MLM.csv", index=False)


def objective(config):

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    (dataset_train, dataset_test, dataset_pred,
     N_fingerprints) = pp.create_datasets(task, dataset, radius, device)
    print('-'*100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    # print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)

    print('Creating a model.')
    torch.manual_seed(1234)
    model = MolecularGraphNeuralNetwork(
            N_fingerprints, config.dim, config.layer_hidden, config.layer_output, device).to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    print('Start training...')

    np.random.seed(1234)

    start = timeit.default_timer()

    best_model = None
    best_score = sys.maxsize
    early_stopping = 0
    for epoch in range(1, iteration+1):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)

        if task == 'classification':
            # prediction_dev = tester.test_classifier(dataset_dev)
            prediction_test = tester.test_classifier(dataset_test)
        if task == 'regression':
            # prediction_dev = tester.test_regressor(dataset_dev)
            prediction_test = tester.test_regressor(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print("Epoch\tTime\tTrain Loss\tTest RMSE")

        if epoch % save_interval == 0:
            os.makedirs("ckpt", exist_ok=True)
            torch.save(model, f"ckpt/epoch_{epoch}_score_{prediction_test}.pth")
        
        if  prediction_test < best_score:
            best_score = prediction_test
            early_stopping = 0
            best_model = deepcopy(model)
            os.makedirs("ckpt", exist_ok=True)
            torch.save(model, f"ckpt/best_rmse_model.pth")
        
        result = f"{epoch}\t{time:.5f}\t{loss_train:.5f}\t{prediction_test:.5f}"

        print(result)

        early_stopping += 1
        if early_stopping > 20:
            print("Early Stopping!")
            break
    return best_score
    

    # print("Predicting...")
    # predict(best_model, dataset_pred)

    # print("FIN")




def main():
    wandb.init(
        entity="bsh",
        project="dacon-ai-drug",
    )

    test_score = objective(wandb.config)

    wandb.log({
        "test_score": test_score
    })



if __name__ == "__main__":

    sweep_configuration = {
        'name': 'HLM',
        'method': 'random',
        'metric': 
            {
                'goal': 'minimize', 
                'name': 'test_score'
            },
        'parameters': 
            {
                # 'radius': {'values': [1, 2, 3]},
                'dim': {'values': [50*i for i in range(1, 5)]},
                'layer_hidden': {'values': [i for i in range(5, 11)]},
                'layer_output': {'values': [i for i in range(5, 11)]},
            }
    }

    task = "regression"
    dataset = "dacon_HLM"
    batch_train = 64
    batch_test = 64
    lr = 1e-3
    lr_decay = 0.99
    decay_interval = 10
    save_interval = 100
    iteration=500
    radius = 1

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="dacon-ai-drug"
    )

    wandb.agent(sweep_id, function=main, count=20)

    
    

