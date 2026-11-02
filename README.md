# Credit Risk Decision Support

![CI](https://github.com/Justin-Fekri/ai-credit-risk-loan-decision-support/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

A tool for loan underwriters. An XGBoost model estimates the chance that a borrower defaults within 12 months, SHAP explains which parts of the application pushed that number up or down, and a set of policy rules turns it into a suggestion: approve, refer, or decline. The underwriter still makes the call and writes down why.

I built it this way on purpose. In lending, a model that just says yes or no isn't very useful, because someone has to be able to explain and defend every decision. So the model here is one piece of evidence, not the decision.

![The review queue. Each file shows its predicted default probability, a risk band, and the suggested action.](docs/risk-desk.png)

## What's in it

- **Model:** a scikit-learn pipeline (one-hot encoding for categories) feeding an XGBoost classifier, trained on income, debt-to-income, credit score, utilization, delinquencies, housing, loan purpose and similar fields.
- **Explanations:** TreeSHAP in probability units. I sum the one-hot columns back into their original field, so "income verification: not verified" shows up as one reason instead of being split across three encoded columns.
- **Policy layer:** separate rules (score, credit score and DTI cut-offs) that map the probability to approve, refer or decline. Keeping them outside the model makes them easy to read and change.
- **Workbench:** a Next.js app where an underwriter opens a file, reads the drivers, and records a decision with notes. It talks to a FastAPI backend.

### Why a file got its score

![SHAP contributions for one applicant. Bars to the right raise the predicted risk, bars to the left lower it, compared with the average applicant.](docs/explanation.png)

## Results

The data is **synthetic**. I wrote a generator that creates 12,000 borrowers with realistic correlations (for example, credit score depends on income, history and debt load) and a default label drawn from a noisy formula. No real consumer data is used.

I held out 2,400 of those records before training:

| Metric | Holdout |
| --- | ---: |
| ROC-AUC | 0.799 |
| PR-AUC | 0.572 |
| F1 at 0.5 | 0.521 |
| Recall | 0.663 |
| Precision | 0.430 |
| Brier score | 0.169 |

Since the test data comes from the same generator, these numbers show the model can recover the patterns I put in. They don't say anything about real applicants. The full confusion matrix, ROC points and subgroup numbers are in `artifacts/metrics.json`.

## Running it

You need Python 3.11+ and Node 20+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.
python scripts/train.py     # writes the model files to artifacts/

python -m uvicorn api.main:app --host 0.0.0.0 --port 45232
```

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Then open http://localhost:45231. The web app forwards `/backend/*` to the API.

Tests and a sample explanation:

```bash
pytest
python scripts/explain_sample.py
```

The fitted model and the review queue aren't committed. `scripts/train.py` creates them, and the API trains them itself if `artifacts/model.joblib` is missing.

## Project layout

```
credit_risk/   data generator, pipeline, SHAP, policy rules
api/           FastAPI: queue, scoring, explanations, decisions
web/           Next.js underwriter workbench
scripts/       training and a sample explanation
artifacts/     evaluation metrics and model card
```

## Model card

![The model card page in the app, with the training data, metrics and subgroup checks.](docs/model-card.png)

- Task: 12-month default, binary
- Model: XGBoost (`hist`) inside a scikit-learn pipeline
- Explanations: `shap.TreeExplainer`, interventional, probability output
- Decision: always a human underwriter; the model never books a loan
- Age is not used as a model input. I only keep it to check whether error rates differ between age groups.

## Limitations

- Synthetic data doesn't look like any real lender's book or approval policy.
- The test set comes from the same generator as the training set, so I can't say how well this would generalize.
- The subgroup checks only cover age bands and employment status, because the synthetic data doesn't include other protected attributes.
- SHAP explains what the model did. It doesn't tell you whether a decision is fair or legal.
- The 0.5 threshold is just a demo setting.
- There's no identity verification, audit log storage, monitoring, or production security.

## License

MIT
