# Updated Development Workflow

## Overall Development Architecture

The project follows a hybrid local/cloud development workflow.

```text
                 Local Laptop
          (VS Code + Codex + Git)

                     │
              Write Python Code
                     │
              Review Generated Code
                     │
                 git commit
                     │
                  git push
                     │
                     ▼

                GitHub Repository
             (Single Source of Truth)

                     ▲
                     │
                  git pull
                     │

        Google Drive Mounted in Colab
                     │
                     ▼

          Persistent Project Directory

                     │
                     ▼

              Google Colab Runtime
         (Training & Experimentation)
```

The repository exists permanently inside Google Drive.

Google Colab accesses that repository directly.

No manual uploading of Python files should be required.

---

# Google Colab Setup

The repository should be cloned **once** into Google Drive.

Example:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content/drive/MyDrive

git clone https://github.com/<username>/thoughtcomm.git
```

This only needs to happen the first time.

For every future Colab session:

```bash
cd /content/drive/MyDrive/thoughtcomm

git pull
```

The project is immediately available.

---

# Development Workflow

Normal development should follow this cycle.

## Step 1

Use VS Code locally.

Codex writes or modifies Python modules.

No work should be performed directly inside Colab.

---

## Step 2

Review all generated code.

Ensure it follows the project architecture.

Run lightweight local tests when possible.

---

## Step 3

Commit changes.

Example:

```bash
git add .

git commit -m "Implement sparse autoencoder"

git push
```

Small, focused commits are preferred over large monolithic commits.

---

## Step 4

Inside Google Colab:

```bash
cd /content/drive/MyDrive/thoughtcomm

git pull
```

The latest project state is now available.

No manual file uploads are necessary.

---

## Step 5

Enable automatic module reloading.

```python
%load_ext autoreload
%autoreload 2
```

Whenever imported Python modules change after a `git pull`, Colab will automatically reload them where possible, reducing the need to restart the runtime.

---

## Step 6

Run experiments through the runner notebook.

The notebook should simply orchestrate execution.

Example:

```python
from experiments.exp02_sparse_autoencoder import run

run()
```

The notebook should never contain reusable implementation logic.

---

# Runner Notebook Policy

The repository should contain a single primary notebook.

```
runner.ipynb
```

Its responsibilities are limited to:

* Mount Google Drive
* Navigate to the repository
* Pull latest changes (optional)
* Install dependencies (if needed)
* Configure experiments
* Launch experiment scripts
* Display results
* Visualize outputs

The notebook should **not** contain:

* model definitions
* loss functions
* utility functions
* reusable training loops
* reusable preprocessing logic

All reusable functionality belongs inside the Python package.

---

# Repository Synchronization

The synchronization strategy is intentionally simple.

Local machine:

```text
Edit Code

↓

Commit

↓

Push
```

Google Colab:

```text
Pull Latest Changes

↓

Run Experiment
```

GitHub is the single source of truth.

Google Drive stores a persistent working copy for Colab.

Google Colab provides temporary compute resources.

---

# Development Cycle

The expected daily workflow is:

```text
Open VS Code

↓

Codex implements feature

↓

Review code

↓

Commit

↓

Push

↓

Open Colab

↓

git pull

↓

%autoreload 2

↓

Run runner.ipynb

↓

Analyze results

↓

Repeat
```

This workflow minimizes manual file transfers while maintaining a clean separation between development and execution.

---

# Guiding Principle

**Code is written locally.**

**Git synchronizes the project.**

**Google Drive provides persistence.**

**Google Colab provides computation.**

Every component has a single responsibility, resulting in a clean, reproducible, and efficient development workflow.
