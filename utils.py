import os
from datetime import datetime


class Logger(object):

    def __init__(self, verbose=0, log_path=None, file_prefix=""):
        self.verbose = verbose
        self.filename = None
        if log_path is not None:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            self.filename = os.path.join(
                log_path, file_prefix + ".log")
            with open(self.filename, "w") as f:
                f.write(self.filename)
                f.write("\n")

    def p(self, s, level=1):
        if self.verbose >= level:
            print(s)
        if self.filename is not None:
            with open(self.filename, "a") as f:
                f.write(datetime.now().strftime("[%m/%d %H:%M:%S]  ") + str(s))
                f.write("\n")

import argparse

parser = argparse.ArgumentParser(description='Main.')
parser.add_argument("--verbose", type=int, default=2)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_trials", type=int, default=10)

# Dataset configuration
parser.add_argument("--path", default="./data")
parser.add_argument("--dataset", default="sota")
parser.add_argument("--num_class", type=int, default=1)
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
parser.add_argument("--model_type", default="corcgcn")
parser.add_argument("--hidden_size", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--clip_output", type=float, default=0.5)

# Training configuration
parser.add_argument("--opt", default="Adam")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=7000)
parser.add_argument("--patience", type=int, default=400)

# Other configuration
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("--result_path", default=None)

args = parser.parse_args()
