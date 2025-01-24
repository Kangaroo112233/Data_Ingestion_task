# Data_Ingestion_task


import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# Example data
y_true = np.array([0, 1, 1, 0, 1, 1, 0])  # True labels
y_probs = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.6, 0.2])  # Predicted probabilities

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

# Interactive Plot Function
def interactive_roc(threshold):
    # Find the index of the closest threshold
    idx = np.argmin(np.abs(thresholds - threshold))
    selected_fpr = fpr[idx]
    selected_tpr = tpr[idx]
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    
    # Highlight the selected threshold point
    plt.scatter([selected_fpr], [selected_tpr], color="red", label=f"Threshold = {threshold:.2f}\nFPR = {selected_fpr:.2f}, TPR = {selected_tpr:.2f}", zorder=5)
    
    # Add labels and legend
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Interactive ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Create a slider for threshold selection
threshold_slider = FloatSlider(
    value=0.5,
    min=min(thresholds),
    max=max(thresholds),
    step=0.01,
    description='Threshold:',
    continuous_update=True
)

# Use `interact` to link the slider to the plot
interact(interactive_roc, threshold=threshold_slider)




Introduction:

In machine learning, especially in critical applications like medical diagnosis, it's essential not only for models to make accurate predictions but also to express their confidence levels appropriately. Calibration refers to how well a model's predicted probabilities align with actual outcomes. For instance, if a model predicts an event with 90% confidence, that event should occur 90% of the time.
Challenges with Current Calibration Measurement:

The Expected Calibration Error (ECE) is a commonly used metric to assess calibration. However, this paper identifies several issues with ECE:
Class Conditioning: ECE often doesn't account for individual classes, potentially overlooking class-specific miscalibrations.
Norm Selection: The choice of norm (e.g., L1 vs. L2) in calculations can influence the assessment of calibration errors.
Probability Binning: Dividing predicted probabilities into bins can introduce biases, especially if not done adaptively based on data distribution.
Comprehensive Empirical Study:

The authors conducted extensive experiments across various datasets, including MNIST, Fashion MNIST, CIFAR-10/100, and ImageNet. They explored different factors affecting calibration measurement:
Evaluating All Probabilities: Instead of focusing solely on the highest predicted probability, assessing all predicted probabilities provides a more comprehensive calibration evaluation.
Thresholding Probability Values: Implementing thresholds can help in distinguishing between confident and less confident predictions.
Class Conditionality: Evaluating calibration on a per-class basis can uncover specific areas where the model may be miscalibrated.
Number of Bins and Adaptive Binning: Choosing the appropriate number of bins and using adaptive binning methods can lead to more accurate calibration assessments.
Norm Selection: Opting for the L2 norm over the L1 norm was found to improve both the optimization for calibration metrics and the consistency of metric evaluations.
Key Findings:

The study revealed that the choice of calibration measurement techniques significantly impacts the evaluation of model calibration. Notably:
Class Conditioning: Evaluating calibration on a per-class basis leads to more effective assessments.
Norm Selection: Using the L2 norm enhances the optimization process for calibration metrics and ensures more consistent evaluations.
Adaptive Binning: Implementing adaptive binning schemes provides stability in metric evaluations, especially when varying the number of bins.
Recommendations:

Based on their findings, the authors suggest:
Conditioning on Class: Always assess calibration on a per-class basis to identify specific areas of miscalibration.
Adaptive Binning: Use binning methods that adapt to the data distribution to ensure accurate calibration measurements.
L2 Norm Usage: Prefer the L2 norm over the L1 norm for more reliable calibration assessments.


Think of calibration like a student's test confidence:

A student predicts they’ll score 80% on a math test.
If the student scores 80% on average whenever they predict this, their prediction is well-calibrated.
For a machine learning model:

If it says, “I’m 90% confident this is a cat,” then out of 100 such predictions, 90 should actually be cats.
If only 60 are cats, the model is not calibrated.
Problems in Measuring Calibration
Ignoring Class-Specific Issues
Example: Imagine a self-driving car model identifies traffic lights and pedestrians.

The model is confident about pedestrians (90%) and is often right.
But for traffic lights, it predicts 90% confidence but is wrong half the time.
If we don’t check calibration for each class (pedestrians vs. traffic lights), we might miss such problems.
Grouping Predictions into Bins
Example: Imagine you have a box of apples and oranges.

You sort them into bins by weight (light, medium, heavy).
If the bins are too big, you might group some apples with oranges by mistake. Similarly, grouping predictions into poorly chosen bins makes calibration less accurate.
Using Wrong Math to Measure
Example: Let’s say you’re calculating your average score using addition instead of proper weighted averages. You’d get a wrong result. Similarly, the paper says some math methods (like L1 norm) are less effective for calibration.

What Does the Paper Suggest?
Check Each Class Separately Example: For our self-driving car, check the predictions for pedestrians and traffic lights separately to find calibration issues.

Use Adaptive Binning Example: Instead of fixed bins (light, medium, heavy), adjust bins based on data. For predictions, this means grouping them intelligently to reflect how often they occur.

Use Better Math (L2 Norm) Example: If you use the right formula, you can get a clearer idea of calibration and improve your predictions.

How Does This Help?
Let’s imagine you’re a student again, trying to decide if a weather app is good:

You record how often it rains when the app predicts 70%, 80%, and 90% confidence.
You check if it rains as often as predicted for each percentage.
If it predicts 70% rain but rains only 50% of the time, the app is poorly calibrated.
With these better methods, you can:

Spot where the app is wrong (e.g., only for 70% predictions).
Fix how the app calculates its confidence.
Similarly, machine learning models can improve by fixing calibration for different situations.

Why Does Calibration Matter?
Medicine: A model predicts the chance of a disease. If poorly calibrated, it might say 90% but be wrong often, leading to wrong treatments.
Self-Driving Cars: If a car is overconfident about recognizing stop signs, it could cause accidents.
Summary with a Fun Analogy
Imagine your friend is guessing their test scores after every exam:

If they guess 80% and actually score 80%, they’re well-calibrated.
If they’re confident but always score lower, their predictions are overconfident.
This paper helps machine learning models avoid being that overconfident friend by measuring and improving their predictions with better tools and methods.


Calibration measures how well a model's predicted probability aligns with reality.

Let’s say a model outputs probabilities 
𝑝
(
𝑥
)
p(x) for an event happening (e.g., whether an image is a cat).
If the model is well-calibrated, then:

𝑃
(
Event happens | Prediction confidence 
=
𝑝
(
𝑥
)
)
≈
𝑝
(
𝑥
)
P(Event happens | Prediction confidence =p(x))≈p(x)
Example:

For 
𝑝
(
𝑥
)
=
0.7
p(x)=0.7, this means that out of 100 such predictions, around 70 should be correct.
2. Expected Calibration Error (ECE)
ECE is a common metric used to measure calibration. It works like this:

Divide the predictions into bins (e.g., 0.0-0.1, 0.1-0.2, ..., 0.9-1.0).
For each bin, calculate:
Accuracy: How many predictions in the bin are correct?
Confidence: Average predicted probability in the bin.
The error for a bin is:

Error
bin
=
∣
Accuracy
bin
−
Confidence
bin
∣
Error 
bin
​
 =∣Accuracy 
bin
​
 −Confidence 
bin
​
 ∣
Finally, ECE is the weighted average error over all bins:

ECE
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
⋅
∣
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
∣
ECE= 
i=1
∑
B
​
  
N
∣B 
i
​
 ∣
​
 ⋅∣Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ∣
Where:

𝐵
𝑖
B 
i
​
 : Bin 
𝑖
i containing predictions.
∣
𝐵
𝑖
∣
∣B 
i
​
 ∣: Number of predictions in bin 
𝑖
i.
𝑁
N: Total number of predictions.
3. Overconfidence and Underconfidence
Overconfident Model: Predicted confidence is higher than actual accuracy.
Example: A model predicts 90% confidence, but only 70% of those predictions are correct.
Underconfident Model: Predicted confidence is lower than actual accuracy.
Example: A model predicts 60% confidence, but 80% of those predictions are correct.
Problems with ECE
The paper discusses limitations in ECE:

A. Class Conditional Calibration
ECE does not account for calibration per class.

Example:

For a self-driving car, predictions for "traffic lights" might be well-calibrated, but predictions for "pedestrians" might not.
Class-wise calibration error:
ECE
class
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
class
⋅
∣
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
∣
ECE 
class
​
 = 
i=1
∑
B
​
  
N 
class
​
 
∣B 
i
​
 ∣
​
 ⋅∣Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ∣
B. Binning Issues
If bins are too coarse, the calibration error might not be accurate. Adaptive binning is suggested, where bins are chosen based on the distribution of predicted probabilities.

Better Calibration Metrics
The paper suggests using the L2 norm instead of the L1 norm for more reliable metrics. Here’s the difference:

L1 Norm (used in ECE):
Calibration Error
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
⋅
∣
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
∣
Calibration Error= 
i=1
∑
B
​
  
N
∣B 
i
​
 ∣
​
 ⋅∣Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ∣
L2 Norm:
Calibration Error
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
⋅
(
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
)
2
Calibration Error= 
i=1
∑
B
​
  
N
∣B 
i
​
 ∣
​
 ⋅(Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ) 
2
 
​
 
The L2 norm gives more weight to large errors, making it better for identifying serious calibration issues.

Simple Example
Predictions:
A model makes the following predictions for being a cat:
70% confident → Correct.
90% confident → Wrong.
50% confident → Correct.
Binning:
Bin 1 (50-60% confidence): Accuracy = 1/1 = 100%.
Bin 2 (70-80% confidence): Accuracy = 1/1 = 100%.
Bin 3 (90-100% confidence): Accuracy = 0/1 = 0%.
ECE Calculation (L1 Norm):
ECE
=
1
3
(
∣
1.0
−
0.5
∣
+
∣
1.0
−
0.7
∣
+
∣
0.0
−
0.9
∣
)
ECE= 
3
1
​
 (∣1.0−0.5∣+∣1.0−0.7∣+∣0.0−0.9∣)
ECE
=
1
3
(
0.5
+
0.3
+
0.9
)
=
0.57
ECE= 
3
1
​
 (0.5+0.3+0.9)=0.57
Conclusion
By understanding and improving calibration using these mathematical tools, models can become more trustworthy and perform better in real-world scenarios, like medical diagnosis, self-driving cars, or weather forecasting.






You said:
Now give me summary of the paper with mathematics
ChatGPT said:
ChatGPT
Summary of the Paper with Mathematics
The paper "Measuring Calibration in Deep Learning" explores how well machine learning models' predicted probabilities match actual outcomes (calibration). It highlights the challenges in current calibration measurement methods and suggests improvements, focusing on more accurate and comprehensive metrics.

1. Calibration Definition
A model is calibrated if the predicted probability 
𝑝
(
𝑥
)
p(x) aligns with the true probability 
𝑃
(
Event happens | Prediction confidence 
=
𝑝
(
𝑥
)
)
P(Event happens | Prediction confidence =p(x)).
For a perfectly calibrated model:

𝑃
(
Event happens | Prediction confidence 
=
𝑝
(
𝑥
)
)
=
𝑝
(
𝑥
)
P(Event happens | Prediction confidence =p(x))=p(x)
Example: If a model predicts 80% confidence, the event should occur 80% of the time when this confidence is predicted.

2. Current Calibration Metrics: Expected Calibration Error (ECE)
ECE measures the difference between a model’s predicted confidence and its actual accuracy. It divides predictions into 
𝐵
B bins and computes:

ECE
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
⋅
∣
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
∣
ECE= 
i=1
∑
B
​
  
N
∣B 
i
​
 ∣
​
 ⋅∣Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ∣
Where:

∣
𝐵
𝑖
∣
∣B 
i
​
 ∣: Number of predictions in bin 
𝑖
i.
𝑁
N: Total number of predictions.
Accuracy
𝐵
𝑖
Accuracy 
B 
i
​
 
​
 : Fraction of correct predictions in bin 
𝑖
i.
Confidence
𝐵
𝑖
Confidence 
B 
i
​
 
​
 : Average predicted confidence in bin 
𝑖
i.
3. Problems with ECE
A. Class-Conditioned Calibration
ECE does not consider calibration per class. For example, if a model predicts "cats" and "dogs," errors specific to one class might be overlooked. Class-specific ECE is:

ECE
class
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
class
⋅
∣
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
∣
ECE 
class
​
 = 
i=1
∑
B
​
  
N 
class
​
 
∣B 
i
​
 ∣
​
 ⋅∣Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ∣
B. Binning Issues
Fixed binning (e.g., dividing confidence into equal intervals) can lead to inaccuracies if predictions are unevenly distributed. Adaptive binning adjusts bin sizes based on the data distribution for better results.

C. Norm Selection
ECE typically uses the L1 norm to calculate errors. However, the paper recommends the L2 norm for better sensitivity to large errors:

L1 Norm (used in ECE):
Calibration Error
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
⋅
∣
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
∣
Calibration Error= 
i=1
∑
B
​
  
N
∣B 
i
​
 ∣
​
 ⋅∣Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ∣
L2 Norm (recommended):
Calibration Error
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
⋅
(
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
)
2
Calibration Error= 
i=1
∑
B
​
  
N
∣B 
i
​
 ∣
​
 ⋅(Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ) 
2
 
​
 
The L2 norm gives more weight to large errors, making it better for identifying significant miscalibration.

4. Proposed Improvements
The paper suggests:

Class-Conditioned Evaluation: Measure calibration separately for each class to catch class-specific issues.
Adaptive Binning: Dynamically adjust bins based on the distribution of predicted probabilities.
L2 Norm: Use the L2 norm to give more importance to large calibration errors.
5. Example of Calibration Measurement
Predictions:
Confidence: [0.6, 0.7, 0.9], Correctness: [1, 1, 0].
Binning:
Bin 1 (50-70%): Confidence = 0.6, Accuracy = 1.0.
Bin 2 (70-90%): Confidence = 0.7, Accuracy = 1.0.
Bin 3 (90-100%): Confidence = 0.9, Accuracy = 0.0.
ECE Calculation (L1 Norm):
ECE
=
1
3
(
∣
1.0
−
0.6
∣
+
∣
1.0
−
0.7
∣
+
∣
0.0
−
0.9
∣
)
ECE= 
3
1
​
 (∣1.0−0.6∣+∣1.0−0.7∣+∣0.0−0.9∣)
ECE
=
1
3
(
0.4
+
0.3
+
0.9
)
=
0.53
ECE= 
3
1
​
 (0.4+0.3+0.9)=0.53
ECE Calculation (L2 Norm):
Calibration Error
=
1
3
(
(
1.0
−
0.6
)
2
+
(
1.0
−
0.7
)
2
+
(
0.0
−
0.9
)
2
)
Calibration Error= 
3
1
​
 ((1.0−0.6) 
2
 +(1.0−0.7) 
2
 +(0.0−0.9) 
2
 )
​
 
Calibration Error
=
1
3
(
0.16
+
0.09
+
0.81
)
=
0.353
≈
0.59
Calibration Error= 
3
1
​
 (0.16+0.09+0.81)
​
 = 
0.353
​
 ≈0.59
6. Why Does This Matter?
Medicine: A calibrated model predicting disease probabilities ensures doctors make decisions based on reliable confidence.
Self-Driving Cars: Properly calibrated models reduce the chance of overconfidence in recognizing objects like pedestrians and traffic signs.
Conclusion
The paper identifies shortcomings in current calibration metrics like ECE and offers practical solutions to improve them. By incorporating class-conditioned metrics, adaptive binning, and the L2 norm, we can better understand and improve the reliability of machine learning models in real-world applications.

A model is calibrated if the predicted probability 
𝑝
(
𝑥
)
p(x) aligns with the true probability 
𝑃
(
Event happens | Prediction confidence 
=
𝑝
(
𝑥
)
)
P(Event happens | Prediction confidence =p(x)).
For a perfectly calibrated model:

𝑃
(
Event happens | Prediction confidence 
=
𝑝
(
𝑥
)
)
=
𝑝
(
𝑥
)
P(Event happens | Prediction confidence =p(x))=p(x)
Example: If a model predicts 80% confidence, the event should occur 80% of the time when this confidence is predicted.

2. Current Calibration Metrics: Expected Calibration Error (ECE)
ECE measures the difference between a model’s predicted confidence and its actual accuracy. It divides predictions into 
𝐵
B bins and computes:

ECE
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
⋅
∣
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
∣
ECE= 
i=1
∑
B
​
  
N
∣B 
i
​
 ∣
​
 ⋅∣Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ∣
Where:

∣
𝐵
𝑖
∣
∣B 
i
​
 ∣: Number of predictions in bin 
𝑖
i.
𝑁
N: Total number of predictions.
Accuracy
𝐵
𝑖
Accuracy 
B 
i
​
 
​
 : Fraction of correct predictions in bin 
𝑖
i.
Confidence
𝐵
𝑖
Confidence 
B 
i
​
 
​
 : Average predicted confidence in bin 
𝑖
i.
3. Problems with ECE
A. Class-Conditioned Calibration
ECE does not consider calibration per class. For example, if a model predicts "cats" and "dogs," errors specific to one class might be overlooked. Class-specific ECE is:

ECE
class
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
class
⋅
∣
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
∣
ECE 
class
​
 = 
i=1
∑
B
​
  
N 
class
​
 
∣B 
i
​
 ∣
​
 ⋅∣Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ∣
B. Binning Issues
Fixed binning (e.g., dividing confidence into equal intervals) can lead to inaccuracies if predictions are unevenly distributed. Adaptive binning adjusts bin sizes based on the data distribution for better results.

C. Norm Selection
ECE typically uses the L1 norm to calculate errors. However, the paper recommends the L2 norm for better sensitivity to large errors:

L1 Norm (used in ECE):
Calibration Error
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
⋅
∣
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
∣
Calibration Error= 
i=1
∑
B
​
  
N
∣B 
i
​
 ∣
​
 ⋅∣Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ∣
L2 Norm (recommended):
Calibration Error
=
∑
𝑖
=
1
𝐵
∣
𝐵
𝑖
∣
𝑁
⋅
(
Accuracy
𝐵
𝑖
−
Confidence
𝐵
𝑖
)
2
Calibration Error= 
i=1
∑
B
​
  
N
∣B 
i
​
 ∣
​
 ⋅(Accuracy 
B 
i
​
 
​
 −Confidence 
B 
i
​
 
​
 ) 
2
 
​
 
The L2 norm gives more weight to large errors, making it better for identifying significant miscalibration.

4. Proposed Improvements
The paper suggests:

Class-Conditioned Evaluation: Measure calibration separately for each class to catch class-specific issues.
Adaptive Binning: Dynamically adjust bins based on the distribution of predicted probabilities.
L2 Norm: Use the L2 norm to give more importance to large calibration errors.
5. Example of Calibration Measurement
Predictions:
Confidence: [0.6, 0.7, 0.9], Correctness: [1, 1, 0].
Binning:
Bin 1 (50-70%): Confidence = 0.6, Accuracy = 1.0.
Bin 2 (70-90%): Confidence = 0.7, Accuracy = 1.0.
Bin 3 (90-100%): Confidence = 0.9, Accuracy = 0.0.
ECE Calculation (L1 Norm):
ECE
=
1
3
(
∣
1.0
−
0.6
∣
+
∣
1.0
−
0.7
∣
+
∣
0.0
−
0.9
∣
)
ECE= 
3
1
​
 (∣1.0−0.6∣+∣1.0−0.7∣+∣0.0−0.9∣)
ECE
=
1
3
(
0.4
+
0.3
+
0.9
)
=
0.53
ECE= 
3
1
​
 (0.4+0.3+0.9)=0.53
ECE Calculation (L2 Norm):
Calibration Error
=
1
3
(
(
1.0
−
0.6
)
2
+
(
1.0
−
0.7
)
2
+
(
0.0
−
0.9
)
2
)
Calibration Error= 
3
1
​
 ((1.0−0.6) 
2
 +(1.0−0.7) 
2
 +(0.0−0.9) 
2
 )
​
 
Calibration Error
=
1
3
(
0.16
+
0.09
+
0.81
)
=
0.353
≈
0.59
Calibration Error= 
3
1
​
 (0.16+0.09+0.81)
​
 = 
0.353
​
 ≈0.59


 import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

###############################################################################
# Helper functions (placeholders for your own versions)
###############################################################################
def compute_ece(y_true, y_prob, num_bins=10):
    """
    Compute a simple Expected Calibration Error (ECE) by:
      1) splitting predictions into 'num_bins' equally sized bins
      2) computing |mean(pred) - fraction_of_positives| in each bin
      3) weighting by the fraction of samples in that bin
    """
    bin_counts, bin_edges = np.histogram(y_prob, bins=num_bins, range=(0,1))
    bin_tot = len(y_prob)

    # fraction of positives per bin & mean predicted probability per bin
    bin_means, _, _ = binned_statistic(y_prob, y_prob, statistic='mean', bins=bin_edges)
    bin_fracs, _, _ = binned_statistic(
        y_prob, y_true, statistic='mean', bins=bin_edges
    )  # fraction_of_positives ~ average of y_true in each bin

    ece_val = 0.0
    for i in range(num_bins):
        count_i = bin_counts[i]
        if count_i > 0:
            gap = abs(bin_means[i] - bin_fracs[i])
            ece_val += (count_i / bin_tot) * gap

    return ece_val

def prepare_reliability_data(y_true, y_prob, num_bins=10):
    """
    Returns two arrays (bin_x, bin_y) for plotting:
      bin_x = the average predicted probability in each bin
      bin_y = the fraction of positives (observed) in that bin
    Uses 'binned_statistic' to find means in each bin.
    """
    # First get bin edges
    bin_counts, bin_edges = np.histogram(y_prob, bins=num_bins, range=(0,1))
    bin_means_pred, _, _ = binned_statistic(
        y_prob, y_prob, statistic='mean', bins=bin_edges
    )
    bin_means_true, _, _ = binned_statistic(
        y_prob, y_true, statistic='mean', bins=bin_edges
    )

    # bin_x is the mean predicted prob in each bin
    # bin_y is fraction_of_positives (the mean of y_true in that bin)
    return bin_means_pred, bin_means_true

###############################################################################
# Core plotting function (matches the screenshot style)
###############################################################################
def plot_reliability_diagram_4i(ax1, ax2, y_true, y_prob,
                                num_bins=10, title_prefix=""):
    """
    Plot a reliability diagram with:
      - ax1 (top) for fraction_of_positives vs predicted prob (line/points)
      - ax2 (bottom) for histogram of predicted probabilities (bars)
    """
    # 1) Compute ECE
    ece_value = compute_ece(y_true, y_prob, num_bins=num_bins)
    
    # 2) Prepare reliability data
    bin_x, bin_y = prepare_reliability_data(y_true, y_prob, num_bins=num_bins)

    # 3) Also get histogram data for the bar plot on ax2
    bin_counts, bin_edges = np.histogram(y_prob, bins=num_bins, range=(0,1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # -- ax1: reliability curve
    ax1.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfect calibration')
    ax1.plot(bin_x, bin_y, marker='o', label='Outputs')
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(f"{title_prefix}\nECE={ece_value:.3f}, bins={num_bins}")
    ax1.legend(loc='best')

    # -- ax2: histogram of predicted probabilities
    # Use width = (bin_edges[1]-bin_edges[0]) so bars touch neatly
    width = bin_edges[1] - bin_edges[0]
    ax2.bar(bin_centers, bin_counts, width=width,
            color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlim([0,1])
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_ylim([0, max(bin_counts)*1.2])
    ax2.grid(False)  # optional

###############################################################################
# Example usage with multiple subplots
###############################################################################
if __name__ == "__main__":

    # Fake data for demonstration
    np.random.seed(0)
    y_true_label = np.random.randint(0, 2, size=500)
    p_platt_label = np.random.uniform(0, 1, size=500)

    y_doc_platt = np.random.randint(0, 2, size=500)
    p_doc_platt = np.random.uniform(0, 1, size=500)

    y_true_iso_label = np.random.randint(0, 2, size=500)
    p_iso_label = np.random.uniform(0, 1, size=500)

    y_doc_iso = np.random.randint(0, 2, size=500)
    p_doc_iso = np.random.uniform(0, 1, size=500)

    # 4 sets of reliability diagrams on a 4x2 grid of axes
    fig, axs = plt.subplots(4, 2, figsize=(10, 15),
                            gridspec_kw={'height_ratios':[3,1,3,1]},
                            sharex='col')
    num_bins = 5

    # 1) Platt (per label)
    plot_reliability_diagram_4i(ax1=axs[0,0], ax2=axs[1,0],
                                y_true=y_true_label,
                                y_prob=p_platt_label,
                                num_bins=num_bins,
                                title_prefix="Platt (per label)")

    # 2) Platt (per doc)
    plot_reliability_diagram_4i(ax1=axs[0,1], ax2=axs[1,1],
                                y_true=y_doc_platt,
                                y_prob=p_doc_platt,
                                num_bins=num_bins,
                                title_prefix="Platt (per doc)")

    # 3) Iso (per label)
    plot_reliability_diagram_4i(ax1=axs[2,0], ax2=axs[3,0],
                                y_true=y_true_iso_label,
                                y_prob=p_iso_label,
                                num_bins=num_bins,
                                title_prefix="Iso (per label)")

    # 4) Iso (per doc)
    plot_reliability_diagram_4i(ax1=axs[2,1], ax2=axs[3,1],
                                y_true=y_doc_iso,
                                y_prob=p_doc_iso,
                                num_bins=num_bins,
                                title_prefix="Iso (per doc)")

    plt.tight_layout()
    plt.show()

