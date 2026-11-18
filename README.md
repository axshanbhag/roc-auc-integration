# Investigating ROC AUC Calculation Integration Methods on Binary Classification Models

Research presented at HTCC. The study compares trapezoidal, midpoint and Simpson's integration on ROC curves for logistic regression, decision trees and random forests.

## Original research artifacts

- [Paper](paper.pdf), Appendix A: reported AUC computations.
- [Paper](paper.pdf#page=11), Appendix B, Listing 4: the Python implementation used for ROC curve generation and numerical integration.

## Run the transcribed listing

`roc_auc_integration.py` is transcribed from Appendix B, Listing 4. The transcription restores indentation, quotation marks, and file names affected by PDF formatting. It preserves the published model settings and integration formulas.

Place the study's `heart_statlog_cleveland_hungary_final.csv` from the Heart Disease Dataset (Comprehensive) in the working directory, then run:

```sh
python -m pip install -r requirements.txt
python roc_auc_integration.py
```

The listing selects the max-heart-rate feature. Change the feature column on line 16 to run the other features from Appendix A. The plot title in the printed listing says Cholesterol even though the configured feature is max heart rate. The original dataset and environment were not used to validate this transcription against the reported results.
