# Investigating ROC AUC Calculation Integration Methods on Binary Classification Models

Research presented at HTCC. The study compares trapezoidal, midpoint and Simpson's integration on ROC curves for logistic regression, decision trees and random forests.

## Original research artifacts

- [Paper](paper.pdf), Appendix A: reported AUC computations.
- [Paper](paper.pdf#page=12), Appendix B, Listing 4: the Python implementation used for ROC curve generation and numerical integration.

The original code is preserved as printed in the paper. A standalone Python source file or notebook has not been located, so this repository cannot yet run the experiment. The study identifies the input as the Heart Disease Dataset (Comprehensive), with the file named `heart_statlog_cleveland_hungary_final.csv`.
