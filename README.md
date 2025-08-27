# MAJOR PROJECT-1: Federated Learning with FedAvg

## 🔍 Ov## 🚀 Quick Start

### Basic Usage

**Windows Command Prompt:**
```batch
REM Activate conda environment
call C:\Users\vicky\miniconda3\condabin\conda.bat activate base
call conda activate fedavg

REM Navigate to FedAvg directory and run
cd FedAv
python fed_avg.py --n_clients=10 --n_epochs=100 --batch_size=32
```

**Linux/Mac/Windows PowerShell:**
```bash
conda activate fedavg
cd FedAv
python fed_avg.py --n_clients=10 --n_epochs=100 --batch_size=32
```This project implements the **Federated Average (FedAvg)** algorithm, a foundational approach in federated learning that enables training machine learning models across decentralized data sources while preserving privacy. The implementation is built using PyTorch and includes support for experiment tracking with Weights & Biases (WandB).

**Key Features:**
- 🔒 Privacy-preserving federated learning
- 📊 Support for both IID and Non-IID data distributions
- 🧠 Multiple model architectures (CNN, MLP)
- 📈 Comprehensive logging and visualization with WandB
- 🧪 Built-in testing and validation frameworks
- 🔄 Flexible client sampling strategies

## 📚 Background

Federated Learning allows multiple parties to collaboratively train a machine learning model without sharing their raw data. The FedAvg algorithm, introduced by McMahan et al., is one of the most widely used methods in federated learning.

**Research Paper:** [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

## 🛠️ Installation

### Prerequisites

- **Python 3.6+**
- CUDA-capable GPU (optional, for acceleration)

### Dependencies

- numpy>=1.22.4
- pytorch>=1.12.0
- torchvision>=0.13.0
- wandb>=0.12.19

### Environment Setup

#### Option 1: Conda (Recommended)

**Linux/Mac:**
```bash
conda env create -f FedAvg/environment.yml
conda activate fedavg
```

**Windows Command Prompt:**
```batch
REM Initialize conda for Windows Command Prompt
call C:\Users\vicky\miniconda3\condabin\conda.bat activate base
call conda activate fedavg
```

**Windows PowerShell:**
```powershell
conda activate fedavg
```

#### Option 2: Pip Installation
```bash
pip install torch torchvision numpy wandb
```

## 🚀 Quick Start

### Basic Usage

Navigate to the FedAvg directory and run:

```bash
cd FedAvg
python fed_avg.py --n_clients=10 --n_epochs=100 --batch_size=32
```

### Advanced Training Example

**Windows Command Prompt:**
```batch
call conda activate fedavg
cd FedAvg
python fed_avg.py ^
    --batch_size=10 ^
    --frac=0.1 ^
    --lr=0.01 ^
    --n_client_epochs=20 ^
    --n_clients=100 ^
    --n_epochs=1000 ^
    --n_shards=200 ^
    --non_iid=1 ^
    --model_name=cnn ^
    --wandb=True
```

**Linux/Mac/Windows PowerShell:**
```bash
conda activate fedavg
cd FedAvg
python fed_avg.py \
    --batch_size=10 \
    --frac=0.1 \
    --lr=0.01 \
    --n_client_epochs=20 \
    --n_clients=100 \
    --n_epochs=1000 \
    --n_shards=200 \
    --non_iid=1 \
    --model_name=cnn \
    --wandb=True
```

## ⚙️ Configuration

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data_root` | String | "../datasets/" | Path to the dataset directory |
| `--model_name` | String | "cnn" | Model architecture (cnn, mlp) |
| `--non_iid` | Int (0/1) | 1 | Data distribution: 0=IID, 1=Non-IID |
| `--n_clients` | Int | 100 | Total number of federated clients |
| `--n_shards` | Int | 200 | Number of data shards for partitioning |
| `--frac` | Float | 0.1 | Fraction of clients participating per round |
| `--n_epochs` | Int | 1000 | Total number of federated rounds |
| `--n_client_epochs` | Int | 5 | Local training epochs per client |
| `--batch_size` | Int | 10 | Training batch size |
| `--lr` | Float | 0.01 | Learning rate |
| `--wandb` | Bool | False | Enable Weights & Biases logging |

## 🏗️ Project Structure

```
MAJOR_PROJECT_1/
├── FedAvg/                    # Main implementation directory
│   ├── data/                  # Data handling modules
│   │   ├── mnist.py          # MNIST dataset wrapper
│   │   ├── sampler.py        # Federated data sampling
│   │   └── __init__.py
│   ├── models/               # Neural network architectures
│   │   ├── models.py         # CNN and MLP implementations
│   │   └── __init__.py
│   ├── tests/                # Unit tests
│   │   ├── test_utils.py     # Utility function tests
│   │   ├── conftest.py       # Test configuration
│   │   └── __init__.py
│   ├── fed_avg.py           # Main FedAvg implementation
│   ├── utils.py             # Helper functions
│   ├── environment.yml      # Conda environment specification
│   ├── sweep.yaml           # WandB hyperparameter sweep config
│   └── pytest.ini          # Testing configuration
├── datasets/                 # Dataset storage (auto-downloaded)
└── README.md                # This file
```

## 🧪 Testing

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
