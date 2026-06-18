# Programming Languages for Data Analysis (CS504) — Final Exam

Final exam for **Programming Languages for Data Analysis (CS504)**, Department of Computer Science,
Bishop's University.

## Face recognition with Logistic Regression (scikit-learn and PyTorch)

Recognising faces in the Olivetti faces dataset (40 people, 10 images each, 64×64 grayscale) using
Logistic Regression, first with scikit-learn and then as a neural network in PyTorch, with analyses
of network depth, optimizer choice, and PCA dimensionality reduction.

The solution is in [`Final_exam.ipynb`](Final_exam.ipynb). The dataset is downloaded automatically
by `sklearn.datasets.fetch_olivetti_faces`.

## What the notebook does

1. Load libraries.
2. Load the Olivetti faces dataset (400 samples, 4096 features, 40 classes).
3. Extract the sample matrix `X` and label vector `y`.
4. Display a random face.
5. Split 70% train / 30% test (samples kept as vectors).
6. **Scikit-learn Logistic Regression**: train with 10-fold cross-validation, evaluate on the test
   set, and show the classification report as a table.
7. **PyTorch Logistic Regression**: a single linear layer (4096 → 40) trained with 10-fold CV, then
   versions with hidden layers, and a bar plot of accuracy vs the number of hidden layers.
8. Bar plot of accuracy vs the optimizer (SGD, Adam, RMSprop, Adagrad).
9. **PCA**: scatter the training set on its first two components, and a bar plot of Logistic
   Regression accuracy as the number of principal components grows from 2 to 10.

## Key results

- Scikit-learn Logistic Regression: 10-fold CV accuracy 0.939, test accuracy 0.975.
- PyTorch Logistic Regression: 10-fold CV accuracy 0.932, test accuracy 0.958.
- Adding hidden layers did not help on this small dataset; the plain linear model was best.
- Logistic Regression accuracy climbed steadily with more PCA components (0.15 at 2 to 0.88 at 10).

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn torch
jupyter notebook Final_exam.ipynb
```

## Files

| File | Description |
|------|-------------|
| `Final_exam.ipynb` | Full solution |
| `FE.pdf` | Exam description |
