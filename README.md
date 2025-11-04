# Federated Learning with FedAvg# Federated Learning with FedAvg# MAJOR PROJECT-1: Federated Learning with FedAvg# MAJOR PROJECT-1: Federated Learning with FedAvg



> A complete implementation of the Federated Average (FedAvg) algorithm for privacy-preserving machine learning across decentralized data.



[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)> Simple implementation of Federated Average algorithm for training machine learning models across decentralized data.

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



---## What is This?[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)## 🔍 Ov## 🚀 Quick Start



## 📖 Table of Contents



- [What is This?](#-what-is-this)Train neural networks on **MNIST**, **CIFAR-10**, and **CIFAR-100** datasets using federated learning - where multiple clients train models on their local data without sharing it.[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

- [Features](#-features)

- [Quick Start](#-quick-start)

- [Installation](#-installation)

- [Running Experiments](#-running-experiments)## Quick Start[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)### Basic Usage

- [Configuration](#-configuration)

- [Troubleshooting](#-troubleshooting)

- [Results](#-expected-results)

- [Citation](#-citation)### Windows



---```batch



## 🔍 What is This?git clone https://github.com/99VICKY99/MAJOR_PROJECT_1.gitA comprehensive implementation of the Federated Average (FedAvg) algorithm for privacy-preserving machine learning across decentralized data sources.**Windows Command Prompt:**



This project implements the **FedAvg algorithm** ([McMahan et al., 2017](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)) - a foundational approach in federated learning that enables training machine learning models across decentralized data sources while preserving privacy.cd MAJOR_PROJECT_1\FedAvg



**Perfect for:**SETUP_EASY.bat```batch

- 🎓 Research in federated learning

- 🔒 Privacy-preserving ML applicationsstart_fedavg_multi_dataset.bat

- 📊 Experiments with non-IID data distributions

- 🧪 Testing federated algorithms```## 🌟 OverviewREM Activate conda environment (Anaconda)



---



## ✨ Features### Linux/Maccall C:\Users\vicky\anaconda3\condabin\conda.bat activate base



✅ **3 Datasets**: MNIST, CIFAR-10, CIFAR-100  ```bash

✅ **3 Data Partitioning Methods**:

   - IID (Independent and Identically Distributed)git clone https://github.com/99VICKY99/MAJOR_PROJECT_1.gitThis project implements the FedAvg algorithm proposed by McMahan et al. (2017) with support for multiple datasets, data partitioning strategies, and modern experiment tracking tools.call conda activate fedavg

   - Shard-based Non-IID

   - Dirichlet Non-IID (α=0.1)  cd MAJOR_PROJECT_1/FedAvg

✅ **2 Model Architectures**: CNN, MLP  

✅ **WandB Integration**: Complete experiment tracking  

✅ **Easy Setup**: One-click installation scripts  

✅ **Cross-Platform**: Windows, Linux, Mac support  # Setup

✅ **18 Pre-configured Experiments**: Just select and run!

conda create -n fedavg python=3.9 -y### Key FeaturesREM Navigate to FedAvg directory and run

---

conda activate fedavg

## 🚀 Quick Start

conda install pytorch torchvision torchaudio -c pytorch -ycd FedAv

### Windows

pip install numpy matplotlib wandb

```batch

git clone https://github.com/99VICKY99/MAJOR_PROJECT_1.git- 📊 **Multiple Datasets**: MNIST, CIFAR-10, CIFAR-100python fed_avg.py --n_clients=10 --n_epochs=100 --batch_size=32

cd MAJOR_PROJECT_1\FedAvg

SETUP_EASY.bat# Run

start_fedavg_multi_dataset.bat

```chmod +x start_fedavg_multi_dataset.sh- 🔄 **Data Partitioning**: IID, Shard-based Non-IID, Dirichlet Non-IID```



### Linux/Mac./start_fedavg_multi_dataset.sh



```bash```- 🧠 **Model Architectures**: CNN, MLP

git clone https://github.com/99VICKY99/MAJOR_PROJECT_1.git

cd MAJOR_PROJECT_1/FedAvg

conda create -n fedavg python=3.9 -y

conda activate fedavg## Features- 📈 **Experiment Tracking**: Weights & Biases (WandB) integration**Linux/Mac/Windows PowerShell:**

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y

pip install numpy matplotlib wandb

chmod +x start_fedavg_multi_dataset.sh

./start_fedavg_multi_dataset.sh✅ 3 Datasets (MNIST, CIFAR-10, CIFAR-100)  - 🖥️ **Cross-Platform**: Windows, Linux, macOS support```bash

```

✅ 3 Data Splits (IID, Shard, Dirichlet)  

That's it! Select an experiment and watch it train. 🎯

✅ 2 Models (CNN, MLP)  - 🚀 **Easy Setup**: One-command installation scriptsconda activate fedavg

---

✅ WandB Logging  

## 📥 Installation

✅ Easy Setup Scripts  cd FedAv

### Prerequisites



- **Anaconda** or **Miniconda**

- **Python 3.9+**## Running Experiments## 📦 Quick Installationpython fed_avg.py --n_clients=10 --n_epochs=100 --batch_size=32

- **NVIDIA GPU** with CUDA 11.6+ (optional, for acceleration)



### Step-by-Step Installation

The launcher provides 18 experiment options. Just select a number:```This project implements the **Federated Average (FedAvg)** algorithm, a foundational approach in federated learning that enables training machine learning models across decentralized data sources while preserving privacy. The implementation is built using PyTorch and includes support for experiment tracking with Weights & Biases (WandB).

#### Windows



**1. Clone the repository:**

```batch```### Windows

git clone https://github.com/99VICKY99/MAJOR_PROJECT_1.git

cd MAJOR_PROJECT_1\FedAvg[1] MNIST - IID

```

[2] MNIST - IID with WandB**Key Features:**

**2. Run automatic setup:**

```batch[3] MNIST - Shard Non-IID

SETUP_EASY.bat

```...and 15 more options```batch- 🔒 Privacy-preserving federated learning



This script will:```

- Create a conda environment named `fedavg`

- Install PyTorch with CUDA support (or CPU-only fallback)git clone https://github.com/99VICKY99/MAJOR_PROJECT_1.git- 📊 Support for both IID and Non-IID data distributions

- Install all dependencies (numpy, matplotlib, wandb)

- Verify the installation## Manual Run



**Wait 5-15 minutes** for completion.cd MAJOR_PROJECT_1\FedAvg- 🧠 Multiple model architectures (CNN, MLP)



**3. Verify installation:**```bash

```batch

call C:\Users\%USERNAME%\anaconda3\condabin\conda.bat activate basepython fed_avg.py --dataset mnist --partition_mode shard --n_epochs 50SETUP_EASY.bat- 📈 Comprehensive logging and visualization with WandB

call conda activate fedavg

python -c "import torch; print('✓ PyTorch:', torch.__version__)"```

```

```- 🧪 Built-in testing and validation frameworks

#### Linux/Mac

### Common Options

**1. Clone the repository:**

```bash- 🔄 Flexible client sampling strategies

git clone https://github.com/99VICKY99/MAJOR_PROJECT_1.git

cd MAJOR_PROJECT_1/FedAvg| Option | Values | Description |

```

|--------|--------|-------------|### Linux/Mac

**2. Create environment:**

```bash| `--dataset` | mnist, cifar10, cifar100 | Choose dataset |

conda create -n fedavg python=3.9 -y

conda activate fedavg| `--partition_mode` | iid, shard, dirichlet | Data split method |## 📚 Background

```

| `--n_epochs` | 50 | Number of training rounds |

**3. Install PyTorch:**

| `--n_clients` | 10 | Number of clients |```bash

**With GPU:**

```bash| `--wandb` | - | Enable logging |

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y

```git clone https://github.com/99VICKY99/MAJOR_PROJECT_1.gitFederated Learning allows multiple parties to collaboratively train a machine learning model without sharing their raw data. The FedAvg algorithm, introduced by McMahan et al., is one of the most widely used methods in federated learning.



**CPU only:**## Project Structure

```bash

conda install pytorch torchvision torchaudio cpuonly -c pytorch -ycd MAJOR_PROJECT_1/FedAvg

```

```

**4. Install other packages:**

```bashFedAvg/**Research Paper:** [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

conda install numpy matplotlib -y

pip install wandb├── data/              # Dataset loaders

```

├── models/            # CNN & MLP models# Create and activate environment

**5. Verify installation:**

```bash├── fed_avg.py         # Main code

python -c "import torch; print('✓ PyTorch:', torch.__version__); print('✓ CUDA available:', torch.cuda.is_available())"

```├── utils.py           # Helper functionsconda create -n fedavg python=3.9 -y## 🛠️ Installation



---├── SETUP_EASY.bat     # Windows setup



## 🧪 Running Experiments└── start_*.bat/.sh    # Run scriptsconda activate fedavg



### Option 1: Interactive Menu (Recommended)```



**Windows:**### Prerequisites

```batch

cd FedAvg## Troubleshooting

start_fedavg_multi_dataset.bat

```# Install dependencies



**Linux/Mac:****"No module named torch"**  

```bash

cd FedAvg```bashconda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y- **Python 3.6+**

chmod +x start_fedavg_multi_dataset.sh

./start_fedavg_multi_dataset.shconda activate fedavg

```

conda install pytorch torchvision -c pytorch -yconda install numpy matplotlib -y- CUDA-capable GPU (optional, for acceleration)

### Available Experiments

```

**MNIST Dataset (Options 1-6):**

- [1] IIDpip install wandb

- [2] IID with WandB logging

- [3] Shard-based Non-IID**"CUDA out of memory"**  

- [4] Shard-based Non-IID with WandB

- [5] Dirichlet Non-IID (α=0.1)```bash```### Dependencies

- [6] Dirichlet Non-IID with WandB

python fed_avg.py --batch_size 5 --n_clients 5

**CIFAR-10 Dataset (Options 7-C):**

- [7-C] Same options as MNIST```



**CIFAR-100 Dataset (Options D-I):**

- [D-I] Same options as MNIST

**More help?** Check [FedAvg/README.md](FedAvg/README.md) for detailed guide.## 🚀 Quick Start- numpy>=1.22.4

Just enter the option number and press Enter!



### Option 2: Manual Command Line

## Results- pytorch>=1.12.0

```bash

conda activate fedavg

cd FedAvg

| Dataset | Method | Epochs | Accuracy |### Windows- torchvision>=0.13.0

# Example: MNIST with shard-based Non-IID

python fed_avg.py --dataset mnist --partition_mode shard --n_epochs 50|---------|--------|--------|----------|



# Example: CIFAR-10 with Dirichlet and WandB| MNIST | IID | 15 | ~98% |```batch- wandb>=0.12.19

python fed_avg.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --n_epochs 60 --wandb --exp_name "my_experiment"

| MNIST | Shard | 50 | ~95% |

# Example: Fast test (5 epochs)

python fed_avg.py --dataset mnist --n_epochs 5| CIFAR-10 | IID | 60 | ~70% |cd FedAvg

```



---

## Citationstart_fedavg_multi_dataset.bat### Environment Setup

## ⚙️ Configuration



### Command-Line Arguments

Based on: [Communication-Efficient Learning of Deep Networks](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)```

| Parameter | Type | Default | Description |

|-----------|------|---------|-------------|

| `--dataset` | str | "mnist" | Dataset: mnist, cifar10, cifar100 |

| `--data_root` | str | "../datasets/" | Path to datasets |## License#### Option 1: Conda (Recommended)

| `--model_name` | str | "cnn" | Model: cnn, mlp |

| `--partition_mode` | str | "shard" | Partitioning: iid, shard, dirichlet |

| `--n_clients` | int | 10 | Number of clients |

| `--n_epochs` | int | 50 | Total federated rounds |MIT License - see [LICENSE](LICENSE)### Linux/Mac

| `--n_client_epochs` | int | 5 | Local training epochs |

| `--batch_size` | int | 10 | Training batch size |

| `--lr` | float | 0.01 | Learning rate |

| `--frac` | float | 1.0 | Fraction of clients per round |---```bash**Linux/Mac/Ubuntu/Debian:**

| `--dirichlet_alpha` | float | 0.1 | Dirichlet alpha parameter |

| `--wandb` | flag | False | Enable WandB logging |

| `--exp_name` | str | - | Experiment name |

**Questions?** Open an [issue](https://github.com/99VICKY99/MAJOR_PROJECT_1/issues) • **Want to help?** See [CONTRIBUTING.md](CONTRIBUTING.md)cd FedAvg```bash

### Recommended Settings



| Dataset | Partition | Epochs | Clients | Batch Size |chmod +x start_fedavg_multi_dataset.sh# Install Anaconda if not already installed

|---------|-----------|--------|---------|------------|

| MNIST | IID | 15 | 10 | 10 |./start_fedavg_multi_dataset.shwget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

| MNIST | Non-IID | 50 | 10 | 10 |

| CIFAR-10 | IID | 60 | 10 | 10 |```bash Anaconda3-2024.02-1-Linux-x86_64.sh

| CIFAR-10 | Non-IID | 60 | 10 | 10 |

| CIFAR-100 | IID | 80 | 10 | 10 |



### Data Partitioning MethodsSelect from 18 pre-configured experiments across three datasets with different partitioning strategies!# Create and activate environment



**1. IID (Independent and Identically Distributed)**conda env create -f FedAvg/environment.yml

- Data randomly distributed among clients

- Each client has samples from all classes## 📖 Documentationconda activate fedavg

- Ideal federated learning scenario

```

**2. Shard-based Non-IID**

- Each client receives exactly 2 shards of different classesFor detailed installation instructions, configuration options, and troubleshooting, see:

- Creates strong class imbalance per client

- Simulates realistic federated scenarios**CentOS/RHEL/Fedora:**



**3. Dirichlet-based Non-IID**📚 **[Complete Documentation](FedAvg/README.md)**```bash

- Label distribution follows Dirichlet(α) distribution

- α=0.1 creates high heterogeneity# Install Anaconda

- More flexible than shard-based partitioning

## 🏗️ Project Structurecurl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

---

bash Anaconda3-2024.02-1-Linux-x86_64.sh

## 🐛 Troubleshooting

```

### 1. "ModuleNotFoundError: No module named 'torch'"

MAJOR_PROJECT_1/# Create and activate environment

**Windows:**

```batch├── FedAvg/                       # Main implementationconda env create -f FedAvg/environment.yml

call conda activate fedavg

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y│   ├── data/                     # Dataset handlers and samplersconda activate fedavg

```

│   ├── models/                   # Neural network architectures```

**Linux/Mac:**

```bash│   ├── tests/                    # Unit tests

conda activate fedavg

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y│   ├── fed_avg.py                # Main FedAvg implementation**Windows Command Prompt:**

```

│   ├── utils.py                  # Helper functions```batch

### 2. "CUDA out of memory"

│   ├── start_fedavg_multi_dataset.bat/.sh  # LaunchersREM Initialize conda for Windows Command Prompt (Anaconda)

**Solution:** Reduce batch size or number of clients:

```bash│   ├── SETUP_EASY.bat            # Windows auto-setupcall C:\Users\vicky\anaconda3\condabin\conda.bat activate base

python fed_avg.py --batch_size 5 --n_clients 5

```│   └── README.md                 # Detailed documentationcall conda activate fedavg



### 3. "conda: command not found" (Linux)│```



**Solution:** Initialize conda:├── datasets/                     # Auto-downloaded datasets

```bash

source ~/anaconda3/etc/profile.d/conda.sh└── README.md                     # This file**Windows PowerShell:**

```

``````powershell

Or add to `~/.bashrc`:

```bashconda activate fedavg

echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc

source ~/.bashrc## 🧪 Available Experiments```

```



### 4. WandB issues

### MNIST (Handwritten Digits)#### Option 2: Pip Installation

**Solution:** Reinstall WandB:

```bash- IID and Non-IID partitioning

pip uninstall wandb -y

pip install wandb- ~98% accuracy achievable**Linux (Ubuntu/Debian):**

wandb login

```- Fast training (~5-15 minutes)```bash



### 5. Dataset not found# Install Python and pip if not available



**Solution:** Datasets download automatically on first run. If it fails, create `../datasets/` directory manually.### CIFAR-10 (10 Object Classes)sudo apt update



---- Natural images datasetsudo apt install python3 python3-pip



## 📊 Expected Results- ~70-75% accuracy achievable



### Performance Benchmarks- Medium training time (~30-60 minutes)# Install dependencies



| Dataset | Partition | Epochs | Accuracy | Training Time |pip3 install torch torchvision numpy wandb

|---------|-----------|--------|----------|---------------|

| MNIST | IID | 15 | ~98% | ~5 min |### CIFAR-100 (100 Classes)```

| MNIST | Shard Non-IID | 50 | ~95-97% | ~15 min |

| MNIST | Dirichlet | 50 | ~90-95% | ~15 min |- Complex classification task

| CIFAR-10 | IID | 60 | ~70-75% | ~30 min |

| CIFAR-10 | Shard Non-IID | 60 | ~65-70% | ~40 min |- ~40-45% accuracy achievable**Linux (CentOS/RHEL/Fedora):**

| CIFAR-100 | IID | 80 | ~40-45% | ~1 hour |

- Longer training time (~1-2 hours)```bash

*Results may vary based on hardware and random initialization*

# Install Python and pip

### WandB Logging

## ⚙️ Example Usagesudo yum install python3 python3-pip  # CentOS/RHEL

To use Weights & Biases for experiment tracking:

# OR

1. **Create account:** [wandb.ai](https://wandb.ai)

2. **Login:** `wandb login````bashsudo dnf install python3 python3-pip  # Fedora

3. **Run with logging:** Add `--wandb --exp_name "your_experiment"`

# MNIST with shard-based Non-IID partitioning

**What gets logged:**

- Training loss per roundpython fed_avg.py --dataset mnist --partition_mode shard --n_epochs 50# Install dependencies

- Test accuracy per round

- Client participation statisticspip3 install torch torchvision numpy wandb

- All hyperparameters

# CIFAR-10 with Dirichlet partitioning and WandB logging```

---

python fed_avg.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --wandb --exp_name "cifar10_experiment"

## 📁 Project Structure

**macOS:**

```

MAJOR_PROJECT_1/# Custom configuration```bash

├── FedAvg/

│   ├── data/python fed_avg.py \# Using Homebrew

│   │   ├── mnist.py              # MNIST dataset handler

│   │   ├── cifar.py              # CIFAR-10/100 handler    --dataset mnist \brew install python3

│   │   └── sampler.py            # Federated data partitioning

│   ├── models/    --partition_mode shard \pip3 install torch torchvision numpy wandb

│   │   └── models.py             # CNN & MLP architectures

│   ├── tests/    --n_clients 10 \```

│   │   ├── test_utils.py         # Unit tests

│   │   └── test_implementation.py    --n_epochs 50 \

│   ├── fed_avg.py                # Main FedAvg implementation

│   ├── utils.py                  # Helper functions    --batch_size 10 \**Windows:**

│   ├── start_fedavg_multi_dataset.bat   # Windows launcher

│   ├── start_fedavg_multi_dataset.sh    # Linux/Mac launcher    --lr 0.01 \```powershell

│   ├── SETUP_EASY.bat            # Windows setup script

│   ├── environment_windows.yml   # Conda environment spec    --model_name cnnpip install torch torchvision numpy wandb

│   └── sweep.yaml                # WandB sweep config

├── datasets/                     # Auto-downloaded datasets``````

└── README.md                     # This file

```



---## 📊 Performance Benchmarks## 🚀 Quick Start



## 🧪 Testing



Run the test suite:| Dataset | Partitioning | Rounds | Accuracy |### Using Provided Scripts (Recommended)



```bash|---------|-------------|--------|----------|

conda activate fedavg

cd FedAvg| MNIST | IID | 15 | ~98% |For easy execution, use the provided startup scripts:

pytest tests/

```| MNIST | Shard Non-IID | 50 | ~95-97% |



Run specific tests:| MNIST | Dirichlet (α=0.1) | 50 | ~90-95% |**Windows:**

```bash

pytest tests/test_implementation.py| CIFAR-10 | IID | 60 | ~70-75% |```batch

pytest tests/test_partitioning.py

```| CIFAR-10 | Shard Non-IID | 60 | ~65-70% |# Navigate to FedAvg directory



---| CIFAR-100 | IID | 80 | ~40-45% |cd FedAvg



## 📚 Citation# Run the Windows batch script (supports MNIST, CIFAR-10, CIFAR-100)



If you use this implementation in your research, please cite:## 🔧 System Requirementsstart_fedavg_multi_dataset.bat



```bibtex```

@inproceedings{mcmahan2017communication,

  title={Communication-efficient learning of deep networks from decentralized data},- **Python**: 3.9 or higher

  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},

  booktitle={Artificial intelligence and statistics},- **RAM**: 8GB minimum, 16GB recommended**Linux/Mac:**

  pages={1273--1282},

  year={2017},- **GPU**: NVIDIA GPU with CUDA 11.6+ (optional, for acceleration)```bash

  organization={PMLR}

}- **Storage**: 5GB for datasets and dependencies# Navigate to FedAvg directory

```

cd FedAvg

**Paper:** [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

## 🐛 Troubleshooting# Make script executable

---

chmod +x start_fedavg_multi_dataset.sh

## 📄 License

### Common Issues# Run the Linux/Mac script

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

./start_fedavg_multi_dataset.sh

---

**"ModuleNotFoundError: No module named 'torch'"**```

## 🤝 Contributing

```bash

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

conda activate fedavgThe scripts provide interactive menus with 18 experiment options:

---

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y- **MNIST Dataset**: IID, Shard, Dirichlet (with/without WandB)

## 🙏 Acknowledgments

```- **CIFAR-10 Dataset**: IID, Shard, Dirichlet (with/without WandB)

- **FedAvg Algorithm**: McMahan et al., Google Research

- **PyTorch**: Facebook AI Research- **CIFAR-100 Dataset**: IID, Shard, Dirichlet (with/without WandB)

- **Weights & Biases**: Experiment tracking platform

**"CUDA out of memory"**

---

```bash### Manual Execution

## 📞 Support

python fed_avg.py --batch_size 5 --n_clients 5

- **Issues**: [GitHub Issues](https://github.com/99VICKY99/MAJOR_PROJECT_1/issues)

- **Repository**: [github.com/99VICKY99/MAJOR_PROJECT_1](https://github.com/99VICKY99/MAJOR_PROJECT_1)```Navigate to the FedAvg directory and run:



---



<div align="center">**WandB not working**```bash



**Built with ❤️ for advancing privacy-preserving machine learning**```bashcd FedAvg



[⬆ Back to Top](#federated-learning-with-fedavg)pip uninstall wandb -ypython fed_avg.py --n_clients=10 --n_epochs=100 --batch_size=32



</div>pip install wandb```


wandb login

```### Advanced Training Example



For more troubleshooting tips, see the [detailed documentation](FedAvg/README.md#troubleshooting).**Windows Command Prompt:**

```batch

## 📚 Research Backgroundcall conda activate fedavg

cd FedAvg

This implementation is based on:python fed_avg.py ^

    --batch_size=10 ^

**McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).**     --frac=0.1 ^

*Communication-Efficient Learning of Deep Networks from Decentralized Data.*     --lr=0.01 ^

Artificial Intelligence and Statistics (AISTATS), pp. 1273-1282.    --n_client_epochs=20 ^

    --n_clients=100 ^

[📄 Read the paper](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)    --n_epochs=1000 ^

    --n_shards=200 ^

### Why Non-IID Can Outperform IID    --non_iid=1 ^

    --model_name=cnn ^

Interestingly, in our experiments, Non-IID partitioning sometimes achieves higher accuracy than IID. This is due to:    --wandb=True

```

1. **Specialization**: Clients become experts on fewer classes

2. **Ensemble Effect**: Server aggregation combines specialized models**Linux/Mac/Windows PowerShell:**

3. **Reduced Interference**: Less gradient conflicts during local training```bash

conda activate fedavg

See our [analysis documentation](FedAvg/analyze_distribution.py) for details.cd FedAvg

python fed_avg.py \

## 🤝 Contributing    --batch_size=10 \

    --frac=0.1 \

Contributions are welcome! Here's how you can help:    --lr=0.01 \

    --n_client_epochs=20 \

1. Fork the repository    --n_clients=100 \

2. Create a feature branch (`git checkout -b feature/amazing-feature`)    --n_epochs=1000 \

3. Commit your changes (`git commit -m 'Add amazing feature'`)    --n_shards=200 \

4. Push to the branch (`git push origin feature/amazing-feature`)    --non_iid=1 \

5. Open a Pull Request    --model_name=cnn \

    --wandb=True

## 📄 License```



This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.## ⚙️ Configuration



## 🙏 Acknowledgments### Command Line Arguments



- **FedAvg Algorithm**: McMahan et al., Google Research| Parameter | Type | Default | Description |

- **PyTorch**: Facebook AI Research|-----------|------|---------|-------------|

- **Weights & Biases**: Experiment tracking and visualization| `--data_root` | String | "../datasets/" | Path to the dataset directory |

- **MNIST, CIFAR**: Classic ML benchmark datasets| `--model_name` | String | "cnn" | Model architecture (cnn, mlp) |

| `--non_iid` | Int (0/1) | 1 | Data distribution: 0=IID, 1=Non-IID |

## 📞 Contact & Support| `--n_clients` | Int | 100 | Total number of federated clients |

| `--n_shards` | Int | 200 | Number of data shards for partitioning |

- **GitHub Issues**: [Report bugs or request features](https://github.com/99VICKY99/MAJOR_PROJECT_1/issues)| `--frac` | Float | 0.1 | Fraction of clients participating per round |

- **Repository**: [github.com/99VICKY99/MAJOR_PROJECT_1](https://github.com/99VICKY99/MAJOR_PROJECT_1)| `--n_epochs` | Int | 1000 | Total number of federated rounds |

| `--n_client_epochs` | Int | 5 | Local training epochs per client |

## 🎯 Roadmap| `--batch_size` | Int | 10 | Training batch size |

| `--lr` | Float | 0.01 | Learning rate |

- [ ] Add more datasets (Fashion-MNIST, SVHN)| `--wandb` | Bool | False | Enable Weights & Biases logging |

- [ ] Implement additional FL algorithms (FedProx, FedNova)

- [ ] Add differential privacy support## 🏗️ Project Structure

- [ ] Create Jupyter notebook tutorials

- [ ] Add model compression techniques```

MAJOR_PROJECT_1/

---├── FedAvg/                              # Main implementation directory

│   ├── data/                            # Data handling modules

<div align="center">│   │   ├── mnist.py                    # MNIST dataset wrapper

│   │   ├── cifar.py                    # CIFAR-10/100 dataset wrapper

**Built with ❤️ for advancing privacy-preserving machine learning**│   │   ├── sampler.py                  # Federated data sampling

│   │   └── __init__.py

[Documentation](FedAvg/README.md) • [Issues](https://github.com/99VICKY99/MAJOR_PROJECT_1/issues) • [Contributing](#contributing)│   ├── models/                         # Neural network architectures

│   │   ├── models.py                   # CNN and MLP implementations

</div>│   │   └── __init__.py

│   ├── tests/                          # Unit tests
│   │   ├── test_utils.py               # Utility function tests
│   │   ├── conftest.py                 # Test configuration
│   │   └── __init__.py
│   ├── fed_avg.py                      # Main FedAvg implementation
│   ├── utils.py                        # Helper functions
│   ├── environment.yml                 # Anaconda environment specification
│   ├── sweep.yaml                      # WandB hyperparameter sweep config
│   ├── start_fedavg_multi_dataset.bat  # Windows startup script (Anaconda)
│   ├── start_fedavg_multi_dataset.sh   # Linux/Mac startup script (Anaconda)
│   └── pytest.ini                      # Testing configuration
├── datasets/                           # Dataset storage (auto-downloaded)
│   ├── MNIST/
│   ├── cifar-10-batches-py/
│   └── cifar-100-python/
└── README.md                           # This file
```

## 🐧 Linux Distribution Notes

### Tested Linux Distributions
- **Ubuntu** 18.04, 20.04, 22.04
- **Debian** 10, 11
- **CentOS** 7, 8
- **Red Hat Enterprise Linux (RHEL)** 7, 8
- **Fedora** 35+

### GPU Support on Linux
For CUDA support on Linux:

**Ubuntu/Debian:**
```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-470
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit
```

**CentOS/RHEL/Fedora:**
```bash
# Enable EPEL repository (CentOS/RHEL)
sudo yum install epel-release
# Install NVIDIA drivers
sudo yum install nvidia-driver cuda-toolkit
```

### Common Linux Issues and Solutions

**Permission Issues:**
```bash
# Make scripts executable
chmod +x FedAvg/start_fedavg.sh

# If conda command not found
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Missing Dependencies:**
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

Run the test suite to validate the implementation:

```bash
cd FedAvg
pytest
```

Specific tests:
```bash
# Test utility functions
pytest tests/test_utils.py

# Test data partitioning
python test_partitioning.py

# Test main implementation
python test_implementation.py
```

## 📊 Hyperparameter Tuning with WandB

This project includes WandB integration for experiment tracking and hyperparameter sweeps.

### Setting up WandB

1. Install WandB: `pip install wandb`
2. Login: `wandb login`
3. Initialize your project: `wandb init`

### Running Hyperparameter Sweeps

To perform a sweep over hyperparameters using WandB:

```bash
wandb sweep FedAvg/sweep.yaml
wandb agent <sweep_id>
```

## 📈 Model Performance

### Supported Models

- **CNN**: Convolutional Neural Network optimized for image classification
  - Target accuracy: 99% on MNIST
  - Suitable for complex feature extraction
  
- **MLP**: Multi-Layer Perceptron
  - Target accuracy: 97% on MNIST  
  - Faster training, good baseline model

### Data Distribution Options

- **IID (Independent and Identically Distributed)**: Data is uniformly distributed across clients
- **Non-IID**: Data is heterogeneously distributed, simulating real-world federated scenarios

## 🔧 Customization

### Adding New Models

1. Implement your model in `FedAvg/models/models.py`
2. Update the model selection logic in `fed_avg.py`
3. Add corresponding tests

### Custom Datasets

1. Create a new dataset class in `FedAvg/data/`
2. Implement the federated sampler for your dataset
3. Update the data loading logic in `fed_avg.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *Artificial Intelligence and Statistics* (pp. 1273-1282).
- [Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/MAJOR-PROJECT-1/issues) section
2. Create a new issue with detailed description
3. Include relevant logs and configuration details

---

**Built with ❤️ for advancing privacy-preserving machine learning**
