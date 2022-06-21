# adversarial_ML
The linearity of classifiers has been postulated to be a cause for adversarial examples. <br>
Normalizing the activation by dividing by the sum of the magnitudes of the inputs enables the output to encode the "confidence" of activation. <br>
This normalization technique is applied to the fully connected layers at the end of a CNN for binary CIFAR10 classifier. <br>
Empirically, it forces successful adversarial attacks to make less localized perturbations.
