# Federated IDS: UNSW-NB15 with Flower & Keras

This project implements an **Intrusion Detection System (IDS)** using **Federated Learning (FL)**. The goal is to train a Deep Learning model collaboratively across multiple nodes without sharing raw, sensitive network traffic data, utilizing the **UNSW-NB15** dataset.

---

## 📊 Dataset: UNSW-NB15

The project utilizes the **UNSW-NB15** dataset, created at the Cyber Range Lab of UNSW Canberra. This dataset simulates a hybrid environment containing modern normal activities and contemporary synthetic attack behaviors.

### Key Characteristics:
* **Raw Traffic:** 100 GB of raw traffic (Pcap files).
* **Total Records:** 2,540,044 records stored across four CSV files.
* **Classes:** Normal and 9 types of attacks (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms).
* **Features:** 49 features generated using tools like Argus and Bro-IDS.

### Data Partitions:
This implementation focuses on the pre-configured partitions:
* `UNSW_NB15_training-set.csv`: 175,341 records.
* `UNSW_NB15_testing-set.csv`: 82,332 records.

---

## 🛠️ Requirements & Installation

1.  **Virtual Environment:** Python 3.12 is recommended.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    Ensure you have your `pyproject.toml` ready and run:
    ```bash
    pip install .
    ```

---

## 🚀 How to Run the Tool

This project uses the Flower **SuperLink** architecture. Run the following commands in separate terminals:



### 1. Start the SuperLink
The central coordinator that manages the federation state.
```bash
flower-superlink --insecure
```

# Federated IDS: UNSW-NB15 with Flower & Keras

This project implements an **Intrusion Detection System (IDS)** using **Federated Learning (FL)**. The goal is to train a Deep Learning model collaboratively across multiple nodes without sharing raw, sensitive network traffic data, utilizing the **UNSW-NB15** dataset.

---

## 📊 Dataset: UNSW-NB15

The project utilizes the **UNSW-NB15** dataset, created at the Cyber Range Lab of UNSW Canberra. This dataset simulates a hybrid environment containing modern normal activities and contemporary synthetic attack behaviors.

### Key Characteristics:
* **Raw Traffic:** 100 GB of raw traffic (Pcap files).
* **Total Records:** 2,540,044 records stored across four CSV files.
* **Classes:** Normal and 9 types of attacks (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms).
* **Features:** 49 features generated using tools like Argus and Bro-IDS.

### Data Partitions:
This implementation focuses on the pre-configured partitions:
* `UNSW_NB15_training-set.csv`: 175,341 records.
* `UNSW_NB15_testing-set.csv`: 82,332 records.

---

## 🛠️ Requirements & Installation

1.  **Virtual Environment:** Python 3.12 is recommended.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    Ensure you have your `pyproject.toml` ready and run:
    ```bash
    pip install .
    ```

---

## 🚀 How to Run the Tool

This project uses the Flower **SuperLink** architecture. Run the following commands in separate terminals:



### 1. Start the SuperLink
The central coordinator that manages the federation state.
```bash
flower-superlink --insecure
```


### 2. Start SuperNodes (Clients)
Launch at least two nodes. Each node represents a local entity (e.g., a different branch of a network) with its own data partition.
```bash
flower-supernode  --insecure  --superlink 127.0.0.1:9092   --clientappio-api-address 127.0.0.1:9094
```

### 3. Run the ServerApp (FL Logic)
This command packages your code into a FAB (Flower App Bundle) and starts the federated training process.
```bash
flwr run . my-federation
```

## ⚙️ Experiment Configuration

You can customize the training parameters in the `[tool.flwr.app.config]` section of your `pyproject.toml`:

| Parameter | Description |
| :--- | :--- |
| **num-server-rounds** | Total number of aggregation rounds. |
| **local-epochs** | Training epochs per client per round. |
| **batch-size** | Local training batch size. |

---

## 📂 Project Structure

* **`fl_ids/server_app.py`**: Server-side logic (FedAvg aggregation and workflow).
* **`fl_ids/client_app.py`**: Client-side logic (Local Keras training and evaluation).
* **`fl_ids/utils/data_loader.py`**: Script for pre-processing and loading the UNSW-NB15 CSV files.
* **`fl_ids/utils/model_loader.py`**: Neural Network architecture (Keras/TensorFlow).

---

## ⚠️ Important Note on FAB Size

If you encounter a **"FAB size exceeds maximum allowed size"** error, ensure your `.gitignore` excludes large CSV files and the `.venv` folder. Flower only needs the source code to run the federation; data should be stored locally on each node.

---
