# Plan 003: Evaluation Protocol and Uncertainty Modeling

## Context & Motivation
Because the forward problem is inherently ambiguous, there are cases where the true phase is genuinely unidentifiable even to a perfect Oracle (e.g., a perfectly flat circular plateau, where it is impossible to tell if it goes up or down).
Instead of silently guessing and failing, a reliable neural network should output its uncertainty. The discrete classification architecture from Plan 002 naturally provides this.

## Steps

1. **Uncertainty Extraction**
   - The softmax outputs from the Sign Head and Wrap Head are probabilities over the discrete hypotheses.
   - Compute the **Shannon Entropy** $H(x,y)$ of the softmax distributions at every pixel to generate an Uncertainty Map.
   - High entropy means the model is explicitly telling the user: "The physics here are too ambiguous to resolve without more priors."

2. **OOD Rejection Threshold**
   - Calibrate an entropy threshold using the validation set.
   - If the mean entropy of an image exceeds the threshold, the system flags the prediction as "Unreliable / Out of Distribution".
   
3. **Rigorous Evaluation Metrics**
   - Stop relying solely on whole-image MAE, as one bad phase wrap destroys the metric even if 95% of the image is perfect.
   - **Metrics to implement:**
     - **Sign Accuracy**: Percentage of pixels where the predicted sign matches GT.
     - **Wrap Accuracy**: Percentage of pixels where the predicted wrap order matches GT.
     - **Residue Count**: Number of topological defects in the final reconstructed phase.
     - **Expected Calibration Error (ECE)**: To verify that when the model is 90% confident, it is correct 90% of the time.

4. **Negative Controls**
   - Create adversarial ambiguous pairs in the test set: Two ground-truth phases $\phi_A$ and $\phi_B$ that produce the *exact same* intensity $I$, but look different.
   - Feed $I$ to the model and verify that the uncertainty entropy spikes at the regions of divergence.

## Verification
- Run evaluation on `Test 2 (OOD)` (from Plan 001). Assert that the mean entropy on Test 2 is statistically significantly higher than on the `Val (IID)` set.
- Assert that Sign Accuracy and Wrap Accuracy are logged in the training loop.
- Generate a qualitative plot showing `(Intensity, GT Phase, Pred Phase, Uncertainty Map)` to visually prove the model highlights ambiguous boundaries.

## Maintenance Notes
- Softmax distributions can be overconfident in modern neural networks. Consider integrating Temperature Scaling or Monte Carlo Dropout if ECE shows the model is poorly calibrated.
