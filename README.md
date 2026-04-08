# Neuro-Fuzzy Disaster Message Routing

This project routes disaster-response messages with a two-stage pipeline:
- a Naive Bayes baseline for the `request` label
- a PyTorch multi-label neural network for intent prediction
- a fuzzy logic layer that turns those intent scores into policy outputs such as `general_aid`, `urgent_medical_response`, and `shelter_supplies`

The main workflow is in `notebooks/03_fuzzy_classifier_.ipynb`.

## Installation

Create and activate a virtual environment, then install the notebook dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install notebook pandas numpy nltk scikit-learn torch
```

## Data

Download the dataset from Kaggle:
- https://www.kaggle.com/datasets/sidharth178/disaster-response-messages

After downloading, create a `data/` folder in the project root and place these files inside it:
- `data/disaster_messages.csv`
- `data/disaster_categories.csv`

## Run

Start Jupyter:

```bash
jupyter notebook
```

Then open and run `notebooks/03_fuzzy_classifier_.ipynb` from top to bottom.

## Results

The strongest outcome of this project is the qualitative routing behavior of the fuzzy policies:
- `general_aid` surfaces broad, actionable requests for help
- `shelter_supplies` prioritizes messages about food, water, tents, and shelter needs
- `urgent_medical_response` highlights more medically oriented or rescue-related messages

In practice, the fuzzy system was most useful as an interpretable and tunable routing layer. The Naive Bayes model remained the stronger direct classifier for the broad `request` label, but the neuro-fuzzy pipeline made it possible to reuse one neural model and apply different response policies without retraining.
