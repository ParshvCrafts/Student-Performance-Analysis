# Student Performance Analysis

An end-to-end machine learning project examining what actually drives student exam outcomes — and
deploying the resulting model as a live web application.

The motivating question is an education-equity one: how much of measured performance is explained by
factors a student controls, versus factors they are born into?

---

## Results

| | |
| :--- | :--- |
| **Model fit** | R² = **0.88** on held-out data |
| **Key finding** | Completing test preparation is associated with a **+7.6 percentage point** improvement |
| **Deployment** | Flask application, packaged for AWS Elastic Beanstalk |

---

## Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `CatBoost` · `Flask` · `AWS Elastic Beanstalk`

---

## Repository layout

```
app.py / application.py     Flask entry points (application.py is the Elastic Beanstalk target)
src/                        Ingestion, transformation, training and prediction pipeline modules
artifacts/                  Serialised model, preprocessor and train/test splits
templates/                  Prediction form and results pages
notebook/                   Exploratory analysis and model comparison
.ebextensions/              Elastic Beanstalk deployment configuration
build.sh                    Build helper
```

---

## Method

1. **Ingest** — load the raw student records and split into train and test sets, persisting both to
   `artifacts/` so every later stage is reproducible.
2. **Transform** — impute and scale numeric features, one-hot encode categoricals, and persist the
   fitted preprocessor alongside the model so training and serving apply identical transformations.
3. **Train** — compare several regressors (linear, tree ensembles, CatBoost) and select on held-out
   R² rather than training fit.
4. **Serve** — expose the selected model through a Flask form that accepts a student profile and
   returns a predicted score.

Separating the preprocessor from the model matters more than it looks: serving skew, where training
and inference transform inputs differently, is the most common way a model that scored well offline
quietly degrades in production.

---

## Running it locally

```bash
git clone https://github.com/ParshvCrafts/Student-Performance-Analysis.git
cd Student-Performance-Analysis

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

Then open `http://127.0.0.1:5000` and submit the prediction form.

---

## Interpreting the results

The associations reported here are **correlational, not causal**. Test preparation completion is not
randomly assigned — students who complete it differ systematically from those who do not, in ways the
dataset does not capture. The +7.6pp figure describes the observed gap, not the effect of assigning
preparation to a random student.

---

## License

MIT
