# Imperial College London 

## MSc AI Dissertation on Agentic AI

This repository contains the full codebase for training, evaluating, and analyzing decision-making agents in text-based interactive environments.
We build on ReAct and StateAct base agents, extending them with retrieval-augmented hinting (RAG), self-generated training data, and LoRA fine-tuning.


Supported environments: ALFWorld (household instruction following), and WebShop (online simulated shopping website).

We combine structured state tracking (StateAct) and lightweight retrieval mechanisms to improve success rates, efficiency, and robustness.


### 🔍 Features

- Base Agents: ReAct (action-only and with reasoning) and StateAct (structured state).

- RAG-based hint retrieval: BM25, TF-IDF, or LLM re-ranking for on-the-fly guidance.

- Self-generated hints: no expert traces required; failures recycled into reusable hint banks.

- Fine-tuning: LoRA/QLoRA training with step-level supervision.

- Multi-backbone support: Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct.

- Evaluation harness: reproducible runs on ALFWorld + WebShop with logging and metrics.

___
### 🛠️ Installation
```
git clone https://github.com/Humaid-Ibrahim/msc-ai-individual-project-agentic-ai-learning.git

cd msc-ai-individual-project-agentic-ai-learning

# create environment and download dependencies
conda env create -f alfworld_environment.yml
conda env create -f webshop_environment.yml
conda env create -f finetune_environment.yml
```

___

### 📂 Repository Structure
```
msc-ai-individual-project-agentic-ai-learning/
├── alfworld_runs/           # scripts + logs for ALFWorld
├── webshop_runs/            # scripts + logs for WebShop
├── alfworld_environment.yml
├── webshop_environment.yml
├── finetune_environment.yml
└── README.md
```

Trained LoRA Adapters can be found [here](https://huggingface.co/LLM-Agents-Imperial).


### Usage
1. Evaluate a baseline agent
```
python3 alfworld_run.py \
    --agent "react" \
    --llm_type "VLLMChat" \
    --model Qwen/Qwen2.5-14B-Instruct \
    --trial_name "react_base" \
    --num_envs 134 \
    --seed 42
```

2. Generate hints from failed trajectories
```
python3 hints_creation.py \
    --csv "game_logs/alfworld_eval_react_base/alfworld_results.csv" \
    --base "/workspace/stateact-ft/alfworld_runs/" \
    --data_root <path to alfworld data> \
    --out "data/hints_react.ndjson" 
```

3. Fine-tune with LoRA

Run the `finetune_alfworld.ipynb` or the `finetune_webshop.ipynb` notebook step-by-step to train the LoRA adapter.


### Running WebShop Server
We provide a pre-built Docker image that runs the WebShop environment server.

This server is required for running agents in the WebShop setting.

```
docker pull docker.io/humaidibrahim/webshop-custom:patched

docker run -it --rm -p 3000:3000 \
    docker.io/humaidibrahim/webshop-custom:patched \
    0.0.0.0 <split> <experiment_name> 3000
```

- 0.0.0.0 → bind to all interfaces (default, change if needed).

- \<split> → dataset split, choose either:

  - train → use training split

  - test → use test split (for evaluation)

- \<experiment_name> → any string, used as a logging folder name (e.g., my_run1).

- 3000 → port (can be changed if needed, must match the -p flag).

\
Example: run on test split

```
docker run -it --rm -p 3000:3000 \
    docker.io/humaidibrahim/webshop-custom:patched \
    0.0.0.0 test evaluation_experiment 3000
```

Example: run on train split

```
docker run -it --rm -p 3000:3000 \
    docker.io/humaidibrahim/webshop-custom:patched \
    0.0.0.0 train training_experiment 3000
```

Once running, the WebShop environment will be accessible at:`http://localhost:3000`. The agent will then interact with it via HTTP requests.
