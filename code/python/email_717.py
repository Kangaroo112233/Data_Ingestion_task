

Following our recent UAT evaluation of the W2 extraction model, I have reviewed field-level performance across all key data points. Based on this, I’ve computed a statistically justified sample size to guide future testing efforts and align with MRM requirements.

🎯 Objective
Ensure that the model evaluation is statistically valid with a high degree of confidence while minimizing manual annotation. We aim to test the model on a dataset that reflects the production distribution and yields reliable accuracy estimates.

📊 UAT Insights
The UAT evaluation of Version 2 of the W2 model reported an overall extraction accuracy of 90.21%, based on 28 fields and several hundred instances. Most high-priority fields (e.g., employee_name, employerName, box1_wage, etc.) exceed 95% accuracy, while a few fields such as box14_other remain performance outliers.

Given the strong model performance, we are using UAT-based accuracy (p = 0.9021) as the estimated population proportion for determining our sample size.

📐 Sample Size Formula
To calculate the minimum statistically significant sample size for proportion estimation, we use:

𝑛
=
𝑍
2
⋅
𝑝
⋅
(
1
−
𝑝
)
𝐸
2
n= 
E 
2
 
Z 
2
 ⋅p⋅(1−p)
​
 
Where:

Z = 1.96 (95% confidence level)

E = 0.05 (±5% margin of error)

p = 0.9021 (model accuracy from UAT)

𝑛
=
(
1.96
)
2
⋅
0.9021
⋅
(
1
−
0.9021
)
(
0.05
)
2
=
3.8416
⋅
0.0884
0.0025
=
0.3396
0.0025
=
135.84
n= 
(0.05) 
2
 
(1.96) 
2
 ⋅0.9021⋅(1−0.9021)
​
 = 
0.0025
3.8416⋅0.0884
​
 = 
0.0025
0.3396
​
 =135.84
✅ Minimum Sample Size Required: ~136 paystubs

📌 Additional Notes
This estimate assumes we’re measuring overall document-level or field-level accuracy using random sampling.

If we need format-specific or field-specific evaluation, we should consider stratified sampling (e.g., 30–50 samples per format or per low-performing field).

For low-confidence fields like box14_other, more focused sampling and error analysis will be required separately.

✅ Recommendation
Proceed with at least 136 annotated paystub documents to ensure statistical validity for model performance estimation with 95% confidence and ±5% error.

Continue stratifying and segmenting for low-accuracy fields to drive targeted improvement.

Let me know if you’d like assistance with scripting the sampling logic or coordinating the golden set annotation process.

Best regards,
