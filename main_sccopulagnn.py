from __future__ import absolute_import, division, print_function
import os
os.environ['JUPYTER_PLATFORM_DIRS'] = '1'
import argparse
import random
import time
from six.moves import cPickle as pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import (to_data, read_sota, read_baron3, read_hk,
  read_tb,read_pbmc)
from models import (MLP, GCN, CopulaModel,
                    RegCopulaGCN, RegCopulaGAT)
from utils import Logger
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import warnings
from focalloss import FocalLoss, MultiClassFocalLossWithAlpha

warnings.filterwarnings('error')

parser = argparse.ArgumentParser(description='Main.')
parser.add_argument("--verbose", type=int, default=2)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_trials", type=int, default=10)

# Dataset configuration
parser.add_argument("--path", default="./data")
parser.add_argument("--dataset", default="human_kidney")
parser.add_argument("--num_class", type=int, default=1)
parser.add_argument("--cross_valid", type=int, default=1, help="cross validation kfolder")
# Synthetic data configuration
parser.add_argument(
    "--lsn_mode", default="daxw",
    help=("Choices: `daxwi`, `xw', or `daxw`. \n"
          "  `daxwi`: only mean is graph-dependent; \n"
          "  `xw`: only cov is graph-dependent; \n"
          "  `daxw`: both mean and cov are graph-dependent."))
parser.add_argument("--num_features", type=int, default=10)
parser.add_argument("--num_nodes", type=int, default=300)
parser.add_argument("--num_edges", type=int, default=5000)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--tau", type=float, default=1.0)

# Model configuration
parser.add_argument("--model_type", default="regcgcn")
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--clip_output", type=float, default=0.5)

# Training configuration
parser.add_argument("--opt", default="AdamW")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=5000)
parser.add_argument("--patience", type=int, default=400)

# Other configuration
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("--result_path", default=None)

args = parser.parse_args()

if __name__ == "__main__":

    for iter_k in range(args.cross_valid):

        save_root = os.path.join("outputs", args.model_type, args.dataset, str(args.num_class))
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        # Set random seed
        if args.seed >= 0:
            args.seed += iter_k
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.device.startswith("cuda"):
                torch.cuda.manual_seed(args.seed)

        # Load data
        data_seed = int(np.ceil(args.seed / float(args.num_trials)))
        if args.dataset == "lsn":
            x, y, adj, datafile = generate_lsn(n=args.num_nodes,
                                            d=args.num_features,
                                            m=args.num_edges,
                                            gamma=args.gamma,
                                            tau=args.tau,
                                            seed=data_seed,
                                            lsn_mode=args.lsn_mode,
                                            root=args.path,
                                            save_file=False)
            data = to_data(x, y, adj=adj)
            data.is_count_data = False
            data.to(args.device)
            
        ###This section is for import sota and baron3 dataset, I also extract the class distribution
        elif args.dataset.startswith("sota"):
            data = read_sota("data")
            #data = load_data_with_negative_sampling("data")
            data.is_count_data = True
            data.to(args.device)
            # Extract training labels using the train_mask
            train_labels = data.y[data.train_mask].cpu().detach().numpy()
            # Check class distribution in the training set
            labels_t, counts_t = np.unique(train_labels, return_counts=True)
            for l, c in zip(labels_t, counts_t):
                print(f"Class {l}: {c} samples, train")
            #test
            test_labels = data.y[data.test_mask].cpu().detach().numpy()
            labels_te, counts_te = np.unique(test_labels, return_counts=True)
            for l, c in zip(labels_te, counts_te):
                print(f"Class {l}: {c} samples, test")
            #valid
            valid_labels = data.y[data.valid_mask].cpu().detach().numpy()
            labels_v, counts_v = np.unique(valid_labels, return_counts=True)
            for l, c in zip(labels_v, counts_v):
                print(f"Class {l}: {c} samples, valid")
        elif args.dataset.startswith("baron3"):
            data = read_baron3("data",seed=data_seed)
            data.is_count_data = True
            data.to(args.device)
            # Extract training labels using the train_mask
            train_labels = data.y[data.train_mask].cpu().detach().numpy()
            # Check class distribution in the training set
            labels, counts = np.unique(train_labels, return_counts=True)
            for l, c in zip(labels, counts):
                print(f"Class {l}: {c} samples")
            #test
            test_labels = data.y[data.test_mask].cpu().detach().numpy()
            labels_te, counts_te = np.unique(test_labels, return_counts=True)
            for l, c in zip(labels_te, counts_te):
                print(f"Class {l}: {c} samples, test")
            #valid
            valid_labels = data.y[data.valid_mask].cpu().detach().numpy()
            labels_v, counts_v = np.unique(valid_labels, return_counts=True)
            for l, c in zip(labels_v, counts_v):
                print(f"Class {l}: {c} samples, valid")
        elif args.dataset.startswith("human_kidney"):
            data = read_hk("data",seed=data_seed)
            data.is_count_data = True
            data.to(args.device)
            # Extract training labels using the train_mask
            train_labels = data.y[data.train_mask].cpu().detach().numpy()
            # Check class distribution in the training set
            labels, counts = np.unique(train_labels, return_counts=True)
            for l, c in zip(labels, counts):
                print(f"Class {l}: {c} samples")
            #test
            test_labels = data.y[data.test_mask].cpu().detach().numpy()
            labels_te, counts_te = np.unique(test_labels, return_counts=True)
            for l, c in zip(labels_te, counts_te):
                print(f"Class {l}: {c} samples, test")
            #valid
            valid_labels = data.y[data.valid_mask].cpu().detach().numpy()
            labels_v, counts_v = np.unique(valid_labels, return_counts=True)
            for l, c in zip(labels_v, counts_v):
                print(f"Class {l}: {c} samples, valid")
        elif args.dataset.startswith("tb"):
            data = read_tb("data",seed=data_seed)
            data.is_count_data = True
            data.to(args.device)
            # Extract training labels using the train_mask
            train_labels = data.y[data.train_mask].cpu().detach().numpy()
            # Check class distribution in the training set
            labels, counts = np.unique(train_labels, return_counts=True)
            for l, c in zip(labels, counts):
                print(f"Class {l}: {c} samples")
            #test
            test_labels = data.y[data.test_mask].cpu().detach().numpy()
            labels_te, counts_te = np.unique(test_labels, return_counts=True)
            for l, c in zip(labels_te, counts_te):
                print(f"Class {l}: {c} samples, test")
            #valid
            valid_labels = data.y[data.valid_mask].cpu().detach().numpy()
            labels_v, counts_v = np.unique(valid_labels, return_counts=True)
            for l, c in zip(labels_v, counts_v):
                print(f"Class {l}: {c} samples, valid")
        elif args.dataset.startswith("pbmc"):
            data = read_pbmc(path='data/pbmc', num_class=args.num_class, seed=data_seed, phase='train')
            data.is_count_data = True
            data.to(args.device)
            # Extract training labels using the train_mask
            train_labels = data.y[data.train_mask].cpu().detach().numpy()
            # Check class distribution in the training set
            labels, counts = np.unique(train_labels, return_counts=True)
            for l, c in zip(labels, counts):
                print(f"Class {l}: {c} samples")
            #test
            test_labels = data.y[data.test_mask].cpu().detach().numpy()
            labels_te, counts_te = np.unique(test_labels, return_counts=True)
            for l, c in zip(labels_te, counts_te):
                print(f"Class {l}: {c} samples, test")
            #valid
            valid_labels = data.y[data.valid_mask].cpu().detach().numpy()
            labels_v, counts_v = np.unique(valid_labels, return_counts=True)
            for l, c in zip(labels_v, counts_v):
                print(f"Class {l}: {c} samples, valid")
            
            ## test
            data_test = read_pbmc(path='data/pbmc', num_class=args.num_class, seed=data_seed, phase='test')
            data_test.is_count_data = True
            data_test.to(args.device)
            # Extract training labels using the train_mask
            train_labels = data_test.y[data_test.train_mask].cpu().detach().numpy()
            # Check class distribution in the training set
            labels, counts = np.unique(train_labels, return_counts=True)
            for l, c in zip(labels, counts):
                print(f"Class {l}: {c} samples")
            #test
            test_labels = data_test.y[data_test.test_mask].cpu().detach().numpy()
            labels_te, counts_te = np.unique(test_labels, return_counts=True)
            for l, c in zip(labels_te, counts_te):
                print(f"Class {l}: {c} samples, test")
            #valid
            valid_labels = data_test.y[data_test.valid_mask].cpu().detach().numpy()
            labels_v, counts_v = np.unique(valid_labels, return_counts=True)
            for l, c in zip(labels_v, counts_v):
                print(f"Class {l}: {c} samples, valid")
        else:
            raise NotImplementedError("Dataset {} is not supported.".format(
                args.dataset))
        print(data)

        ###change the evaluation matric
        def evaluate_classification_metrics(preds, labels, evals):
            metrics = {}

            # change type
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            if isinstance(evals, torch.Tensor):
                evals = evals.cpu().numpy()   

            try:
                #accuracy
                metrics['Accuracy'] = accuracy_score(labels, preds)
            except ValueError as e:
                print(f"Could not calculate Accuracy: {e}")

            try:
                #precision
                metrics['Precision'] = precision_score(labels, preds, average='weighted', zero_division=1)
            except ValueError as e:
                print(f"Could not calculate Precision: {e}")

            try:
                # recall
                metrics['Recall'] = recall_score(labels, preds, average='weighted', zero_division=1)
            except ValueError as e:
                print(f"Could not calculate Recall: {e}")

            try:
                # F1 score
                metrics['F1 Score'] = f1_score(labels, preds, average='weighted', zero_division=1)
            except ValueError as e:
                print(f"Could not calculate F1 Score: {e}")
            
            try:
                # PR-AUC
                precision, recall, _ = precision_recall_curve(labels, evals)
                f1, lr_auc = f1_score(labels, preds), auc(recall, precision)
                metrics['PRAUC'] = lr_auc
            except ValueError as e:
                print(f"Could not calculate PRAUC Score: {e}")

            try:
                # ROC 
                fpr, tpr, _ = roc_curve(labels, evals)
                roc_auc = auc(fpr, tpr)
            except Exception as e:  # Catching general exceptions for simplicity
                    print(f"Error during ROC computation: {e}")
            metrics["ROC_AUC"]=roc_auc


            #roc curve
            plt.figure(figsize=(12, 8))
            plt.plot(fpr, tpr, lw=2,
                            label='ROC curve (area = %0.2f)' % roc_auc)
                        #label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            ###title and file name need to be changed based on label
            plt.title(f'Receiver Operating Characteristic for class {args.num_class}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_root,f"ROC for {args.num_class}.png"))

            plt.close()

            plt.figure(figsize=(12, 8))
            precision, recall, _ = precision_recall_curve(labels, evals)
            f1, lr_auc = f1_score(labels, preds), auc(recall, precision)
            # plot the precision-recall curves
            no_skill = len(labels[labels==1]) / len(labels)
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='1')
            plt.plot(recall, precision, marker='.', label='pr curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            ###title and file name need to be changed based on label
            plt.title(f'Precision-Recall Curves for class {args.num_class}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_root,f"PR for {args.num_class}.png"))
            plt.close()
            return metrics

        minimize_metric = -1

        # if not data.is_count_data:
        marginal_type = "Poisson"
        # else:
        #     marginal_type = "Binomial"

        # Log file
        time_stamp = time.time()
        log_file = (
            "data__{}__model__{}__lr__{}__h__{}__seed__{}__stamp__{}").format(
            args.dataset, args.model_type, args.lr, args.hidden_size, args.seed,
            time_stamp)
        if args.dataset == "lsn":
            log_file += "__datafile__{}".format(os.path.splitext(datafile)[0])
        log_path = os.path.join(args.path, "logs")
        lgr = Logger(args.verbose, log_path, log_file)
        lgr.p(args)

        # Model config
        model_args = {
            "num_features": data.x.size(1),
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "activation": "relu"
        }

        if args.model_type in ["corcgcn", "regcgcn", "corcsage", "regcsage","copula", 'regcgat', 'corcgat']:
            model_args["marginal_type"] = marginal_type

        if args.model_type == "mlp":
            model = MLP(**model_args)
        elif args.model_type == "gcn":
            model = GCN(**model_args)
        elif args.model_type == "regcgcn":
            model = RegCopulaGCN(**model_args)
        elif args.model_type == "copula":
            model = CopulaModel(**model_args)
        else:
            raise NotImplementedError("Model {} is not supported.".format(
                args.model_type))
        model.to(args.device)

        # Optimizer
        if args.opt == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        elif args.opt == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
        elif args.opt == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
        else:
            raise NotImplementedError("Optimizer {} is not supported.".format(
                args.opt))

        # Training objective
        if hasattr(model, "nll"):
            def train_loss_fn(model, data):
                return model.nll(data)

        else:
        ###
        #In here I delete other loss function, I also changed the original to
        #Binary Cross Entropy and it seems works as well. 
        #If we are dealing with binary data, maybe we could go with BCE?
            # loss_fn = nn.CrossEntropyLoss()
            loss_fn = MultiClassFocalLossWithAlpha([0.01, 4], gamma=6)
            def criterion(logits, labels):
                #return torch.mean(torch.exp(logits) - labels * logits)
                return loss_fn(logits, labels.long())
            # criterion = nn.BCELoss()
        
            def train_loss_fn(model, data):
                return criterion(
                    model(data)[data.train_mask], data.y[data.train_mask])
                #criterion = nn.BCELoss()

        # Training and evaluation
        def train():
            model.train()
            optimizer.zero_grad()
            loss = train_loss_fn(model, data)
            loss.backward()
            optimizer.step()

        def export_for_r(evals, y, file_name):
            # Create a DataFrame
            df = pd.DataFrame({
                'Evals': evals,
                'Y': y    })
            # Export to CSV
            csv_file_name = os.path.join(save_root, f"data_{args.num_class}.csv")
            df.to_csv(file_name, index=False)

        def test():
            model.eval()
            performance_results = {}
            with torch.no_grad():
                if hasattr(model, "predict"):
                      preds,evals = model.predict(data,num_samples=1000) 
                      export_for_r(evals.cpu(), data.y.cpu(), f'output_{args.model_type}.csv')
                      ###The num_samples could be changed maybe?
                    # Evaluate performance
                      train_metric = evaluate_classification_metrics(preds[data.train_mask], data.y[data.train_mask], evals[data.train_mask])
                      valid_metric = evaluate_classification_metrics(preds[data.valid_mask], data.y[data.valid_mask],evals[data.valid_mask])
                      test_metric = evaluate_classification_metrics(preds[data.test_mask], data.y[data.test_mask],evals[data.test_mask])

                      performance_results = {
                        'train': train_metric, 
                        'valid': valid_metric, 
                        'test': test_metric
                       }

                else:
                    preds = model(data)
                    if marginal_type == "Poisson":
                        preds = torch.exp(preds)
                if args.clip_output != 0:  # clip output logits to avoid extreme outliers
                    left = torch.min(data.y[data.train_mask]) / args.clip_output
                    right = torch.max(data.y[data.train_mask]) * args.clip_output
                    preds = torch.clamp(preds, left, right)
            return performance_results
                
        patience = args.patience
        best_metric = np.inf
        stats_to_save = {"args": args, "traj": []}
        for epoch in range(args.num_epochs):
            train()
            loss = train_loss_fn(model,data)
            #loss = nn.BCELoss()
            if (epoch + 1) % args.log_interval == 0:
                performance_results = test()
        # Averaging metrics across labels
                print(performance_results)
                valid_accuracy = performance_results['valid']['Accuracy']
                test_accuracy = performance_results['test']['Accuracy']
                train_accuracy = performance_results['train']['Accuracy']

                valid_auc = performance_results['valid']['ROC_AUC']
                test_auc = performance_results['test']['ROC_AUC']
                train_auc = performance_results['train']['ROC_AUC']

                valid_prauc = performance_results['valid']['PRAUC']
                test_prauc = performance_results['test']['PRAUC']
                train_prauc = performance_results['train']['PRAUC']
                
                this_metric = valid_accuracy * minimize_metric
                patience -= 1
                if this_metric < best_metric:
                    patience = args.patience
                    best_metric = this_metric
                    stats_to_save["valid_metric"] = valid_accuracy
                    stats_to_save["test_metric"] = test_accuracy
                    stats_to_save["valid_auc"] = valid_auc 
                    stats_to_save["test_auc"] = test_auc
                    stats_to_save["valid_prauc"] = valid_prauc 
                    stats_to_save["test_prauc"] = test_prauc
                    stats_to_save["epoch"] = epoch
                    stats_to_save["loss"] = loss
                stats_to_save["traj"].append({
                    "epoch": epoch,
                    "valid_metric": valid_accuracy,
                    "test_metric": test_accuracy,
                    "valid_auc": valid_auc,
                    "test_auc": test_auc,
                    "valid_prauc": valid_prauc,
                    "test_prauc": test_prauc,
                    "loss":loss
                })
                if patience == 0:
                    break
                lgr.p("Epoch {}: train {:.4f}, valid {:.4f}, test {:.4f},valid_auc: {:.4f}, test_auc {:.4f}, valid_prauc: {:.4f}, test_prauc {:.4f}, Loss: {:.4f}".format(
                    epoch, train_accuracy, valid_accuracy, 
                    test_accuracy, valid_auc, test_auc, valid_prauc, test_prauc, loss))
        lgr.p("-----\nBest epoch {}: valid {:.4f}, test {:.4f}".format(
            stats_to_save["epoch"], stats_to_save["valid_metric"],stats_to_save["valid_auc"],
            stats_to_save["test_metric"],stats_to_save["test_auc"]))
        df = pd.DataFrame(stats_to_save["traj"])
        # Save to Excel file
        excel_filename = os.path.join(save_root, f"result_{args.num_class}.xlsx")
        df.to_excel(excel_filename, index=False)
        print(f"Results saved to {excel_filename}")

        # Assuming stats_to_save["traj"] stores the trajectory of metrics across epochs
        epochs = [entry["epoch"] for entry in stats_to_save["traj"]]
        valid_metrics = [entry["valid_metric"] for entry in stats_to_save["traj"]]
        test_metrics = [entry["test_metric"] for entry in stats_to_save["traj"]]

        # Extracting AUC scores
        valid_auc_scores = [entry["valid_auc"] for entry in stats_to_save["traj"]]
        test_auc_scores = [entry["test_auc"] for entry in stats_to_save["traj"]]

        loss = [entry["loss"] for entry in stats_to_save["traj"]]

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, valid_auc_scores, label='Validation AUC')
        plt.plot(epochs, test_auc_scores, label='Test AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC Score')
        plt.title('AUC Score vs. Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_root, f'AUC vs Epochs,{args.num_class}.png'))
        plt.close()

        loss_value = [x.detach().cpu().numpy() for x in loss] 
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss_value, label='Loss')
        plt.title('Epoch vs Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_root, f"Loss-{args.num_class}.png"))





        # Write outputs
        if args.verbose == 0:
            if args.result_path is None:
                result_path = os.path.join(args.path, "results")
            else:
                result_path = args.result_path
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            result_file = (
                "data__{}__valid__{}__test__{}__model__{}__lr__{}__h__{}__seed__{}"
                "__stamp__{}").format(
                args.dataset, stats_to_save["valid_metric"],
                stats_to_save["test_metric"], args.model_type, args.lr,
                args.hidden_size, args.seed, time_stamp)
            if args.dataset == "lsn":
                result_file += "__datafile__{}".format(os.path.splitext(datafile)[0])
            with open(os.path.join(result_path, result_file), "wb") as f:
                pickle.dump(stats_to_save, f)
