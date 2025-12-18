# RelationExtraction

---

## Task Description
The dataset consists of questions about movies and actors, such as *“When was Toy Story released?”*. Each question is associated with a label representing the intent, for example, `movie.initial_release_date`. The objective was to train a model that could correctly classify new user queries based on the information they seek.

The data was split into **training**, **validation**, and **test** sets. The model was trained on the training set, tuned using the validation set, and evaluated on the test set only after finalizing model decisions to ensure fair evaluation.

---

## Training Procedure
During training, the model was moved to the appropriate device (CPU or GPU). The **Adam optimizer** was used along with the **BCEWithLogitsLoss** criterion, which is suitable for binary and multi-label classification tasks.

The model was trained for up to **500 epochs**, but training typically stopped earlier due to an **early stopping mechanism**. In each epoch:
- The model iterated over the training data
- Performed forward and backward passes
- Updated weights using the optimizer
- Tracked and stored training loss

After each epoch, the model was evaluated on the validation set using `torch.no_grad()`. Validation loss and the **weighted F1 score** were computed. The weighted F1 score was used as the primary metric for:
- Saving the best-performing model checkpoint
- Resetting or incrementing a patience counter

If the validation F1 score did not improve for **10 consecutive epochs**, training was stopped early. After training completed, the best saved model weights were restored, and final validation performance was reported. Losses and F1 scores were saved for later analysis and visualization.

---

## Hyperparameter Experiments

### Learning Rate
I experimented with learning rates of **0.01**, **0.001**, and **0.0001**. The best performance was achieved with a learning rate of **0.001**, yielding a validation F1 score of **0.89**. The other learning rates resulted in poor or unstable validation performance, suggesting that:
- **0.01** was too aggressive and likely caused unstable updates
- **0.0001** was too small and led to underfitting

---

### Dropout Rate
Dropout was used as a regularization technique to prevent overfitting. I tested dropout rates of **0.1**, **0.3**, and **0.7**. All three produced similar results, but **0.3** achieved the highest validation F1 score. This aligns with expectations, as:
- Lower dropout (0.1) may lead to overfitting
- Higher dropout (0.7) may prevent the model from learning effectively

---

### Number of Layers
I also varied the number of hidden layers, testing **2**, **3**, and **4** layers. Models with **2 and 3 layers** performed similarly on validation and test sets, while the **4-layer model** achieved the highest test F1 score but the lowest validation F1 score. This suggests potential overfitting when increasing model depth beyond what the task requires.
