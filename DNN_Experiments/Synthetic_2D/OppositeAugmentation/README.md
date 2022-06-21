# Data Augmentation Defense
State-of-the-art defensive techniques against L_2 attacks augment the training dataset with adversarial examples (adversarial training). <br>
However, this type of training is inherently computationally expensive and unstable, requiring additional hyperparameters to be carefully tuned. <br>
My proposed method seeks to avoid the expensive computation of adversarial examples during training. <br>
Each training batch is augmented by moving each sample a certain distance toward a randomly selected sample of the opposite class. <br>
This can potentially act as an approximation for an L_2 adversarial example.
