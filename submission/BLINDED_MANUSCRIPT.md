# External validation, conformal uncertainty, and feature stability in machine-learning epigenetic clocks trained on established methylation panels

**Article type:** Research article

## Abstract

DNA methylation clocks are often treated as a setting in which flexible machine-learning models should improve chronological-age prediction. We tested that assumption using two public whole-blood Gene Expression Omnibus cohorts and a 418 cytosine-phosphate-guanine site union of established clock probes. Models were trained in one cohort and externally tested in the other, then reversed. When trained on GSE40279 and tested on GSE87571, a plain elastic-net baseline reached mean absolute error 3.85 years, while ElasticNetCV reached 3.95 years; their paired bootstrap confidence interval for the mean absolute error difference crossed zero. The unweighted ensemble had mean absolute error 4.21 years and was distinguishably worse than the best linear model. In the reverse direction, ElasticNetCV reached mean absolute error 3.79 years and was distinguishably better than gradient boosting, multilayer perceptron, plain elastic net, and the ensemble. Split-conformal intervals based on ElasticNetCV were empirically near nominal in one direction, with coverage 0.898 for nominal coverage 0.90, but undercovered in the reverse direction, with coverage 0.748. SHapley Additive exPlanations analysis of the gradient-boosting model identified cg16867657 as rank 1 in both directions, while broader feature ranking was weakly stable across all 418 sites. These findings support a reality-check conclusion: on established methylation-clock panels, added model complexity did not improve external accuracy, and uncertainty calibration did not transfer symmetrically across cohorts.

**Keywords:** DNA methylation; epigenetic clock; external validation; elastic net; conformal prediction; SHAP

## Introduction

DNA methylation age predictors have become a central measurement technology in gerontology because they convert high-dimensional methylation profiles into age-related scalar quantities. Early clocks showed that chronological age could be recovered from methylation at selected CpG sites in saliva, blood, and multiple tissues [1-3]. Subsequent work extended these predictors from chronological-age estimation toward healthspan, mortality, and pace-of-aging applications [4-9]. The result is a fast-moving literature in which epigenetic clocks are used not only to estimate age, but also to study disease risk, environmental exposure, tissue specificity, and geroscience interventions [10-20].

The field now contains several overlapping model-building traditions. One tradition emphasizes interpretable penalized linear regression, especially elastic-net regression, which was used in influential methylation clocks and remains attractive when the number of CpG predictors is large relative to cohort size [2,3,8,12,36]. Another tradition applies more flexible machine-learning models, including tree ensembles and neural networks, to capture nonlinear methylation-age structure [13,35,37]. Recent studies have also broadened clock design for biological interpretability, cell-type specificity, technical platform effects, use across tissues or assay platforms, and links to functional phenotypes [15-18,21-32]. These developments are useful, but they also create a recurring practical question: when the feature set is already constrained to established clock CpGs, does additional model complexity improve external predictive performance?

The answer cannot be assumed from model class alone. A nonlinear model has greater representational capacity, but capacity is useful only if the additional structure is shared between training and test cohorts. In methylation-clock work, shared structure may be partly biological and partly technical. Whole-blood methylation profiles reflect methylation changes within cell types, shifts in cell-type composition, assay processing, batch effects, and cohort ascertainment. A model can use all of these signals to predict chronological age in one cohort, but only a subset may transfer to another cohort. This is why an apparently better within-cohort fit can be less useful than a simpler model that travels better across cohorts.

That question matters because methylation-clock results are often carried across cohorts. A model that looks accurate within a training cohort may fail when array processing, cell composition, cohort recruitment, sex distribution, disease burden, or age composition changes. Several reviews and methodological papers have cautioned that epigenetic clocks require careful validation before individual-level or translational interpretation [5,19,26,27]. The concern is particularly acute for biological-age "age gaps", defined as predicted age minus chronological age. Age gaps are frequently used as downstream phenotypes, but an age gap can reflect biology, technical transfer, model misspecification, residual cohort structure, or a mixture of these factors. External validation is therefore not a secondary check; it is part of the measurement claim.

A second gap concerns uncertainty. Most clock papers report point predictions and aggregate error metrics, but individual predictions are rarely accompanied by calibrated prediction intervals. Conformal prediction offers a distribution-free framework for converting any point predictor into prediction sets with finite-sample coverage guarantees under exchangeability [33,34]. In biomedical aging research, this is appealing because an individual predicted age without uncertainty can invite overinterpretation. However, conformal validity depends on the calibration data being exchangeable with the future test data. Cross-cohort methylation transfer is exactly the setting in which that assumption can be strained. It is therefore useful to test not only whether conformal intervals can be constructed, but whether their coverage transfers in both cohort directions.

A third gap concerns interpretability stability. Feature attribution methods such as SHAP can summarize the CpGs that drive a flexible predictor [35]. In methylation clocks, a stable top CpG may point to well-known age-associated biology, while unstable lower-ranked CpGs may reveal cohort-specific model behavior. ELOVL2 methylation is a useful reference point because it has been repeatedly identified as a strong age-associated methylation marker [10,11]. Still, high attribution for one CpG does not imply that a whole attribution ranking is stable. For a clock to support mechanistic interpretation, it is important to distinguish a stable anchor from an unstable tail of feature rankings.

The interpretability issue is sharper for aging biomarkers than for many ordinary prediction tasks. A CpG that contributes to chronological-age prediction may be tempting to describe as an aging mechanism. That inference is not warranted from prediction alone. Age-associated methylation can arise from developmental timing, immune remodeling, exposure history, cellular composition, stochastic drift, selection, or technical platform behavior [4,5,17,20-23,28]. For this reason, a feature-attribution analysis should be asked a narrower question: are the same CpGs used by the model when the training and test cohorts are reversed? A stable attribution pattern would not prove mechanism, but an unstable one would weaken broad biological interpretation.

This study addresses these gaps as a reality-check analysis rather than as a proposal of a new clock. The analysis uses two public Gene Expression Omnibus cohorts, GSE40279 and GSE87571, and restricts features to the 418-CpG union of established Hannum and Horvath clock probes. The goal is not to maximize leaderboard performance by searching a wide feature space. The goal is to ask whether common machine-learning increases in complexity improve cross-cohort accuracy when the CpG panel is already biologically and historically selected.

The study was guided by the following research questions. First, under bidirectional external validation, does a regularized linear model match or outperform more complex machine-learning models trained on the same methylation-clock CpG panel? Second, can split-conformal intervals provide nominal individual-level uncertainty coverage when a model is trained and calibrated in one cohort and tested in another? Third, are SHAP-derived CpG importances stable across the two external-validation directions, or is stability limited to a small number of anchor CpGs? Fourth, do epigenetic age gaps from a predictor trained without sex as an input show consistent sex associations across transfer directions?

The central finding is deliberately modest. Elastic-net models matched or outperformed histogram gradient boosting, this multilayer-perceptron implementation, and an unweighted ensemble in external validation. Split-conformal intervals were empirically near nominal in one direction but undercovered in the reverse direction. SHAP analysis identified cg16867657 as a stable dominant CpG, consistent with the known ELOVL2 aging signal, but broader rankings were weakly stable. Together, these results support a parsimony-first interpretation of methylation-clock machine learning on established CpG panels.

This framing is intentionally different from a clock-development paper that presents a new predictor as the main product. The analysis asks how far one can get with a simple model, how much uncertainty remains at the individual level, and how stable the apparent biological drivers are when the external-validation direction changes. Negative answers to these questions are useful because they define the boundary between measurement signal and model ambition.

## Materials and Methods

### Study design and data sources

This was a secondary analysis of public, de-identified DNA methylation data from the Gene Expression Omnibus. Two GEO-derived CSV files were used: `data/gse40279_clock_cpgs.csv` and `data/gse87571_clock_cpgs.csv`. These reduced CSV files were generated from real GEO beta values and retained only established clock CpGs. No simulated, synthetic, or fabricated samples were used. The analysis was generated by `make_clock.py` from the two pre-existing data files in `data/`.

The GEO source files were `GSE40279_series_matrix.txt.gz` for GSE40279 and, for GSE87571, `GSE87571_series_matrix.txt.gz` for sample metadata plus the supplementary beta matrices `GSE87571_matrix1of2.txt.gz` and `GSE87571_matrix2of2.txt.gz`. GSE40279 beta values were read from the GEO series matrix. GSE87571 beta values were read from the two supplementary matrix files; bare `Xn` beta columns were retained and paired `Xn.1` columns were not used. The clock-CpG lists came from bio-learn v0.9.1: `Hannum.csv` and `Horvath1.csv`. The reduced files were retrieved/generated on 2026-07-01.

The data audit identified 656 raw rows in GSE40279 and 732 raw rows in GSE87571. Rows missing chronological age were excluded because chronological age was the supervised target. After this exclusion, GSE40279 contributed 656 usable rows and GSE87571 contributed 729 usable rows; this is the source of the 732 raw versus 729 analyzed GSE87571 counts. GSE40279 had 0 missing beta cells, 0 missing age cells, and 0 missing sex cells. GSE87571 had 10 missing beta cells, 3 missing age cells, and 2 missing sex cells before age exclusion. After age exclusion, GSE40279 spanned ages 19-101 years with 338 female and 318 male samples, and GSE87571 spanned ages 14-94 years with 388 female and 341 male samples. The feature set contained 418 CpG beta-value features. Sex was included as an additional model input for the main point-prediction models when available.

The design emphasized external validation. Models were trained on all usable GSE40279 rows and tested on all usable GSE87571 rows. The direction was then reversed, training on GSE87571 and testing on GSE40279. This bidirectional design avoids treating either cohort as the only development cohort and makes it possible to observe transfer asymmetry.

One feature-selection caveat is important. GSE40279 is the original Hannum-clock discovery cohort [2]. Therefore, "external validation" in this study refers to model fitting, prediction, calibration, and attribution transfer between cohorts, not to the historical selection of every CpG in the fixed feature panel. Because all model classes received the same 418-CpG union of Hannum and Horvath probes, the model-class comparison remains internally valid, but the fixed panel should not be described as feature-selection-naive with respect to GSE40279.

### Preprocessing

Preprocessing was fitted inside the training split only. CpG beta values were median-imputed using the training split only, then standardized. Sex was most-frequent imputed using the training split only and one-hot encoded. The same fitted preprocessing pipeline was then applied to the corresponding external test cohort. This procedure was used to avoid leakage of test-cohort distributional information into training-time imputation or scaling.

All metrics were calculated on external test predictions after preprocessing and model fitting were completed within the training cohort. The primary accuracy metrics were mean absolute error, root mean squared error, Pearson correlation between chronological age and predicted age, and median absolute error. Mean absolute error was treated as the main error metric because it is directly interpretable in years and is less dominated by extreme errors than root mean squared error.

### Models

The model set was chosen to contrast parsimony with flexible machine learning while keeping the feature set fixed. The first model was ElasticNetCV, a regularized linear model with internal cross-validation over elastic-net penalty settings. The second was a plain elastic-net baseline with fixed regularization settings. The third was histogram gradient boosting, a nonlinear tree-based ensemble. The fourth was a multilayer perceptron. The fifth was an unweighted ensemble average of ElasticNetCV, histogram gradient boosting, and multilayer perceptron predictions. The ensemble was a simple voting regressor rather than a stacked meta-model, because the goal was to avoid adding another training layer on modest cohort sizes.

The models were not interpreted as competing clock products. They were treated as probes of a practical modeling question: when the CpG feature panel is already constrained to established clock CpGs, is flexible modeling necessary for external chronological-age prediction?

The plain elastic-net baseline was included as a parsimony check. ElasticNetCV can adapt regularization through internal cross-validation, while the plain elastic-net baseline fixes the regularization form. If both linear models perform similarly, the result argues that much of the externally transferable signal is already captured by a simple regularized linear map. Histogram gradient boosting and multilayer perceptron were included to test whether nonlinear structure adds transferable signal. The unweighted ensemble was included because averaging can sometimes reduce model-specific error. A failure of the ensemble to beat the best base learner would therefore be informative, not incidental.

No model was selected or tuned using the external test cohort. All reported external metrics are out-of-cohort metrics. This distinction is central to the study because the research question concerns cohort transfer, not within-cohort optimization.

All hyperparameters were fixed in code before external testing. The random seed was 20260703. ElasticNetCV used l1_ratio values 0.1, 0.3, 0.5, 0.7, and 0.9; 30 alpha values on `logspace(-2, 1.5, 30)`; five-fold internal cross-validation; maximum 20,000 iterations; and `n_jobs=1`. The plain elastic-net baseline used alpha = 1.0, l1_ratio = 0.5, maximum 20,000 iterations, and the same random seed. Histogram gradient boosting used squared-error loss, learning rate 0.04, 350 maximum boosting iterations, 15 maximum leaf nodes, L2 regularization 0.02, early stopping enabled, and the same random seed. The multilayer perceptron used hidden layers of 64 and 16 units, ReLU activation, alpha = 0.001, initial learning rate 0.001, maximum 500 iterations, early stopping enabled, validation fraction 0.15, 25 iterations without improvement before stopping, target standardization through `TransformedTargetRegressor`, and the same random seed. The unweighted ensemble averaged ElasticNetCV, histogram gradient boosting, and multilayer-perceptron predictions.

### Paired bootstrap model comparison

For each external-validation direction, the best linear model was defined as the lower-MAE member of the ElasticNetCV and plain elastic-net baseline pair in that direction. Paired bootstrap confidence intervals were then calculated for the mean absolute error difference between each other model and that best linear reference model. The resampling unit was the external-test sample, preserving the pairing of absolute errors across models. Differences were defined as comparison-model mean absolute error minus reference-model mean absolute error, so positive values indicate that the comparison model had larger error. The analysis used 1000 bootstrap resamples and seed 20260703.

### Within-cohort cross-validation

Within-cohort performance was evaluated with 10-fold cross-validation separately in each cohort. Cross-validation was used as a reference for how model performance appeared when training and testing occurred inside the same cohort distribution. These results were not used as the primary evidence of generalization. The primary inference came from the two external-validation directions.

The within-cohort analysis served two purposes. First, it checked whether each model could learn a chronological-age signal when train and test folds came from the same cohort. Second, it provided a contrast against external validation. A model that performs well in cross-validation but loses accuracy in the external cohort is not necessarily a poor model; it may be a model that learned cohort-specific structure. For this reason, cross-validation was interpreted as a diagnostic, not as the principal benchmark.

### Split-conformal intervals

Split-conformal prediction was applied primarily to ElasticNetCV, the recommended regularized linear model. The same split-conformal procedure was also applied to the unweighted ensemble as a secondary comparison. In each training cohort, the data were split into a proper-training subset and a calibration subset using a 0.20 calibration fraction and seed 20260703. The point predictor was fitted on the proper-training subset. Absolute prediction errors on the calibration subset were used as nonconformity scores. For nominal coverage 0.90, the finite-sample conformal quantile was calculated and then added to and subtracted from each external test prediction to form a symmetric prediction interval in years.

For the GSE40279 to GSE87571 direction, the proper-training size was 524 and the calibration size was 132. For the GSE87571 to GSE40279 direction, the proper-training size was 583 and the calibration size was 146. Empirical coverage was calculated as the fraction of external test chronological ages falling between the lower and upper conformal interval limits. Mean interval width, median interval width, point mean absolute error, point root mean squared error, point Pearson correlation, and point median absolute error were also recorded.

The conformal analysis was deliberately evaluated out of cohort. The calibration subset came from the same cohort as the proper-training subset, while the test set came from the other cohort. This is a demanding use case. It asks whether calibration errors learned inside one cohort are appropriate for another cohort. The usual split-conformal finite-sample guarantee does not hold across non-exchangeable external cohorts; if exchangeability is strained, empirical coverage can depart from nominal coverage even when the conformal algorithm is implemented correctly.

### SHAP feature-attribution analysis

SHAP analysis was applied to the histogram gradient-boosting model because it is nonlinear and benefits from post hoc attribution. For each external-validation direction, the gradient-boosting model was fitted in the training cohort, transformed training data were used as background data, and transformed external test data were explained. The SHAP background and explained samples were capped at 200 rows each, using seeds 20260703 and 20260704 for sampling when a cohort exceeded that cap. Mean absolute SHAP values were calculated for CpG features. Feature rankings were then compared between the two transfer directions.

Stability was quantified in two ways. First, Spearman rank correlation was calculated across all 418 CpGs. Second, the overlap of the top 20 CpGs in each direction was summarized with a Jaccard index and an overlap count. The top CpGs were also ranked by the average of their mean absolute SHAP values across the two directions. The Spearman p value was treated as descriptive only because CpG methylation features are correlated and the rank test does not convert attribution ranks into independent biological evidence.

### Age-gap sex association

Epigenetic age gap was defined as predicted age minus chronological age. To avoid a circular sex association, age-gap analyses used external ElasticNetCV predictions from a second set of models trained without sex as a predictor. For each transfer direction, rows with nonmissing binary sex were retained. Mean age gap was summarized overall and by sex. The female-minus-male age-gap difference was calculated. Raw association was summarized with Pearson correlation between a female indicator and age gap and with a Welch test comparing female and male age gaps. Age-adjusted association was summarized with a linear regression of age gap on female indicator and chronological age; the female coefficient and p value were reported, along with the corresponding partial correlation. These analyses were exploratory because the primary goal of the study was model transfer, not sex-difference inference.

The age-gap analysis was included because age gaps are common downstream quantities in methylation-clock studies. It was not intended to establish a sex-specific biological aging claim. Instead, it tested whether a simple downstream association would be stable when the training and test cohorts were reversed. Instability in this analysis would reinforce the need to validate age-gap associations under the same transfer conditions used to validate point prediction.

### Figures and tables

Five figures and three tables were generated from the analysis outputs. Figure 1 shows ElasticNetCV predicted age versus chronological age in both external-validation directions. Figure 2 shows external-validation mean absolute error by model and transfer direction. Figure 3 shows the top 20 CpGs by average mean absolute SHAP value. Figure 4 shows primary ElasticNetCV split-conformal coverage against nominal 0.90 coverage. Figure 5 shows no-sex ElasticNetCV external-test age-gap distributions. Table 1 summarizes external-validation performance by model and transfer direction. Table 2 summarizes split-conformal calibration. Table 3 summarizes SHAP feature-stability metrics.

## Results

### Data audit and external-validation setup

The final analysis used 656 usable GSE40279 samples and 729 usable GSE87571 samples. GSE87571 contained 732 raw rows, but 3 rows missing chronological age were excluded before supervised learning. The feature matrix contained 418 CpG beta-value features, with sex included as an additional covariate in the main preprocessing pipeline. Missingness was limited. GSE40279 had 0 missing beta cells, 0 missing age cells, and 0 missing sex cells. GSE87571 had 10 missing beta cells, 3 missing age cells, and 2 missing sex cells before age exclusion. After age exclusion, the external-validation sample sizes were 729 for testing on GSE87571 and 656 for testing on GSE40279.

This setup produced a strict cross-cohort test. In one direction, every model was trained on GSE40279 and tested on GSE87571. In the reverse direction, every model was trained on GSE87571 and tested on GSE40279. The same model classes and preprocessing logic were used in both directions.

The two directions were not treated as interchangeable replicates. They were treated as separate transfer experiments. This matters because the training cohort determines imputation values, scaling parameters, fitted coefficients or nonlinear structure, and conformal calibration scores. Reversing the direction can therefore change both the point predictor and the uncertainty wrapper, even though the same two datasets are involved.

### Regularized linear models matched or outperformed more complex models

The main external-validation result was that model complexity did not improve prediction accuracy. Table 1 shows the full external-validation metric set by model and transfer direction. When trained on GSE40279 and tested on GSE87571, the plain elastic-net baseline had the lowest mean absolute error (3.85 years), with ElasticNetCV close behind (3.95 years). In the reverse direction, ElasticNetCV had the lowest mean absolute error (3.79 years). The more flexible models did not improve on these linear baselines in either direction.

**Table 1. External-validation performance by model and transfer direction.**

| Model | Direction | MAE (years) | RMSE (years) | Median AE (years) | Pearson r |
|---|---|---:|---:|---:|---:|
| Plain elastic-net baseline | GSE40279 -> GSE87571 | 3.85 | 4.69 | 3.32 | 0.981 |
| ElasticNetCV | GSE40279 -> GSE87571 | 3.95 | 4.93 | 3.31 | 0.982 |
| Histogram gradient boosting | GSE40279 -> GSE87571 | 5.99 | 7.26 | 5.18 | 0.974 |
| MLP | GSE40279 -> GSE87571 | 7.91 | 10.19 | 6.14 | 0.897 |
| Ensemble average | GSE40279 -> GSE87571 | 4.21 | 5.18 | 3.60 | 0.972 |
| Plain elastic-net baseline | GSE87571 -> GSE40279 | 4.41 | 5.75 | 3.59 | 0.944 |
| ElasticNetCV | GSE87571 -> GSE40279 | 3.79 | 5.11 | 3.07 | 0.945 |
| Histogram gradient boosting | GSE87571 -> GSE40279 | 4.76 | 6.23 | 3.78 | 0.913 |
| MLP | GSE87571 -> GSE40279 | 11.29 | 14.18 | 9.66 | 0.744 |
| Ensemble average | GSE87571 -> GSE40279 | 4.95 | 6.38 | 4.08 | 0.908 |

Figure 1 visualizes the external ElasticNetCV predictions against chronological age. The scatterplots show that the recommended linear model retained a clear age signal in both transfer directions. Figure 2 compares all models and clarifies that the ensemble was not the accuracy winner. The best-performing approaches were regularized linear models. This was not a small interpretive detail; it was the central performance result.

The ensemble result is important because it rules out a common compromise interpretation. One might expect that averaging a strong linear model, a tree ensemble, and this multilayer-perceptron implementation would protect against the weaknesses of any single model. That expectation did not hold here. In the GSE40279 to GSE87571 direction, the ensemble mean absolute error was 4.21 years, higher than both ElasticNetCV and the plain elastic-net baseline. In the GSE87571 to GSE40279 direction, the ensemble mean absolute error was 4.95 years, higher than ElasticNetCV, the plain elastic-net baseline, and histogram gradient boosting. The average inherited enough error from weaker base learners to lose the parsimony advantage.

The paired bootstrap analysis qualified the linear-model comparison. In the GSE40279 to GSE87571 direction, the best linear reference was the plain elastic-net baseline. ElasticNetCV had 0.10 years higher mean absolute error, but the 95% paired bootstrap confidence interval crossed zero (-0.07 to 0.27 years), so this analysis does not provide evidence that the two linear models differed in that direction. The ensemble, histogram gradient boosting, and multilayer perceptron were distinguishably worse than the best linear reference, with mean absolute error differences of 0.36 years (95% CI 0.15 to 0.57), 2.14 years (95% CI 1.85 to 2.45), and 4.06 years (95% CI 3.56 to 4.56), respectively. In the GSE87571 to GSE40279 direction, ElasticNetCV was the best linear reference. The plain elastic-net baseline, histogram gradient boosting, ensemble, and multilayer perceptron were distinguishably worse, with mean absolute error differences of 0.63 years (95% CI 0.46 to 0.79), 0.97 years (95% CI 0.72 to 1.24), 1.17 years (95% CI 0.85 to 1.46), and 7.50 years (95% CI 6.80 to 8.18), respectively.

Within-cohort cross-validation was more optimistic for several models, particularly in GSE87571. In GSE40279, ElasticNetCV had 10-fold mean absolute error 3.43 years with standard deviation 0.40 years, root mean squared error mean 4.60 years, Pearson correlation mean 0.948, and median absolute error mean 2.79 years. In GSE87571, ElasticNetCV had 10-fold mean absolute error 2.70 years with standard deviation 0.28 years, root mean squared error mean 3.55 years, Pearson correlation mean 0.985, and median absolute error mean 2.20 years. These cross-validation results confirm that the models learned strong age signals within each cohort, but they also illustrate why within-cohort performance is insufficient for the main claim. The external validation results are the more relevant test of cohort transfer.

### Split-conformal intervals were empirically near nominal in one direction but undercovered in the reverse direction

Primary split-conformal intervals were constructed for ElasticNetCV predictions at nominal coverage 0.90. As summarized in Table 2, empirical coverage was near nominal when models were trained on GSE40279 and tested on GSE87571, but coverage fell below nominal in the reverse direction. For ElasticNetCV, coverage was 0.898 with mean interval width 15.09 years in the GSE40279 to GSE87571 direction. In the reverse direction, coverage was 0.748 with mean interval width 11.08 years. The corresponding calibration sizes were 132 and 146, and the external test sizes were 729 and 656.

**Table 2. Split-conformal calibration summary.**

| Predictor | Role | Direction | Empirical coverage (nominal 0.90) | Mean interval width (years) | Calibration n | Test n |
|---|---|---|---:|---:|---:|---:|
| ElasticNetCV | Primary | GSE40279 -> GSE87571 | 0.898 | 15.09 | 132 | 729 |
| ElasticNetCV | Primary | GSE87571 -> GSE40279 | 0.748 | 11.08 | 146 | 656 |
| Ensemble average | Secondary comparison | GSE40279 -> GSE87571 | 0.904 | 17.31 | 132 | 729 |
| Ensemble average | Secondary comparison | GSE87571 -> GSE40279 | 0.776 | 17.56 | 146 | 656 |

Figure 4 summarizes the primary ElasticNetCV coverage asymmetry. The result is not that conformal prediction failed generally. In one direction, observed coverage was empirically near the target. The result is that conformal calibration did not transfer symmetrically across cohorts. This matters for clock deployment because conformal prediction intervals are sometimes described as model-agnostic uncertainty wrappers. They are model-agnostic in construction, but their finite-sample guarantee depends on exchangeability. In this external-cohort setting, that assumption should not be assumed.

The interval widths also require practical interpretation. Primary ElasticNetCV mean widths of 15.09 years and 11.08 years are wide relative to the point prediction errors of the best linear models. For individual-level interpretation, a clock prediction accompanied by such an interval communicates uncertainty more honestly than a single point estimate. However, wide intervals may also limit the practical value of a single individual estimate if the intended use is fine-grained ranking or intervention response.

The undercoverage direction is particularly informative. For ElasticNetCV, the reverse-direction interval was narrower and coverage was lower, indicating that the calibration error distribution from GSE87571 did not adequately represent the GSE40279 external test errors. The secondary ensemble comparison produced wider intervals and slightly higher reverse-direction coverage, but it remained below nominal. For methylation-clock reporting, this supports a practice of presenting both interval width and empirical external coverage rather than asserting calibration from the conformal algorithm alone.

### SHAP identified a stable ELOVL2-linked anchor but weak broader stability

SHAP analysis of the gradient-boosting model showed one dominant CpG. The top-ranked CpG by average mean absolute SHAP value was cg16867657, with average mean absolute SHAP 7.93. It ranked first in both external-validation directions. Its mean absolute SHAP value was 8.10 for train GSE40279 and test GSE87571, and 7.76 for train GSE87571 and test GSE40279 (Table 3). The prominence of cg16867657 is consistent with the well-established age association of ELOVL2 methylation [10,11].

The next ranked CpGs were much smaller in attribution magnitude. The second-ranked CpG, cg06639320, had average mean absolute SHAP 1.30 and ranked third in both directions. The third-ranked CpG, cg10501210, had average mean absolute SHAP 1.24 and ranked fifth in one direction and second in the other. The fourth-ranked CpG, cg19283806, had average mean absolute SHAP 1.04 and ranked eighth in one direction and fourth in the other. The fifth-ranked CpG, cg22454769, had average mean absolute SHAP 0.80 and ranked twelfth in one direction and fifth in the other.

The full ranking was not strongly stable. Across all 418 CpGs, Spearman rank correlation between the two SHAP rankings was 0.290 (p = 1.6 x 10^-9). The p value is of limited interpretive value because CpG methylation features are correlated and the rank test is not a test of independent biological mechanisms. The top-20 Jaccard index was 0.18, and the two top-20 sets overlapped by 6 CpGs (Table 3). Figure 3 shows the top 20 CpGs by average mean absolute SHAP value, but the stability metrics are necessary for interpretation. The figure alone might suggest a ranked biological hierarchy. The bidirectional stability analysis instead supports a more cautious conclusion: cg16867657 is a stable anchor, while much of the broader attribution tail is cohort-sensitive.

**Table 3. SHAP feature-stability summary.**

| Contrast | Spearman rho across 418 CpGs | Spearman p value | Top-20 Jaccard | Top-20 overlap count | Top CpG rank | Top CpG mean-abs-SHAP |
|---|---:|---:|---:|---:|---|---|
| GSE40279 -> GSE87571 vs GSE87571 -> GSE40279 | 0.290 | 1.6 x 10^-9 | 0.18 | 6 | cg16867657: 1 / 1 | 8.10 / 7.76 (average 7.93) |

This distinction affects how feature-attribution results should be used. In this analysis, SHAP was informative for identifying a dominant CpG and for demonstrating instability beyond that dominant signal. It does not support strong mechanistic claims for every highly ranked CpG. A feature can receive high attribution because it is biologically linked to aging, because it is a proxy for cohort structure, because it interacts with preprocessing and model form, or because it is useful in one transfer direction but not the other.

The top-20 overlap gives a concrete measure of that instability. A 6-CpG overlap between the two top-20 sets means that most CpGs appearing in a top-20 list in one direction did not appear in the top-20 list in the other direction. This does not make those CpGs unimportant in every context. It means their apparent importance was conditional on the training cohort and transfer direction. The result argues for reporting rank-stability metrics alongside attribution plots, especially when feature-attribution results are used to motivate biological interpretation.

### Age-gap sex associations were direction-dependent

External age gaps from ElasticNetCV models trained without sex were summarized by sex. In the GSE40279 to GSE87571 direction, 729 rows had nonmissing binary sex, including 388 female and 341 male rows. The mean age gap was 2.91 years. The female mean age gap was 2.77 years, and the male mean age gap was 3.07 years. The female-minus-male difference was -0.30 years. The raw Pearson correlation between female indicator and age gap was -0.038 with p = 0.312. After adjustment for chronological age, the female coefficient was -0.29 years with p = 0.332, and the partial correlation was -0.036.

In the GSE87571 to GSE40279 direction, 656 rows had nonmissing binary sex, including 338 female and 318 male rows. The mean age gap was -2.09 years. The female mean age gap was -2.78 years, and the male mean age gap was -1.36 years. The female-minus-male difference was -1.42 years. The raw Pearson correlation between female indicator and age gap was -0.147 with p = 0.000151. After adjustment for chronological age, the female coefficient was -1.27 years with p = 0.000176, and the partial correlation was -0.146.

Figure 5 shows the no-sex ElasticNetCV age-gap distributions for the two transfer directions. The sex association was not consistent across directions. One direction showed little evidence of a sex-associated age-gap difference after age adjustment, while the reverse direction showed a larger female-minus-male difference with a lower p value. These results should be interpreted as exploratory. They are useful mainly as a warning that downstream age-gap associations can depend on transfer direction and model behavior even when the predictor itself is trained without sex.

## Discussion

This study produced a negative result in the sense most relevant to applied methylation-clock modeling: more complex machine-learning models did not improve external age prediction on the established clock-CpG panel. Elastic-net models were the most reliable performers across the two transfer directions. When trained on GSE40279 and tested on GSE87571, the plain elastic-net baseline had the lowest mean absolute error, 3.85 years, and ElasticNetCV was close, 3.95 years; the paired bootstrap confidence interval for that 0.10-year difference crossed zero. When trained on GSE87571 and tested on GSE40279, ElasticNetCV had the lowest mean absolute error, 3.79 years, and the paired bootstrap intervals supported lower error than the plain elastic-net baseline, histogram gradient boosting, this multilayer-perceptron implementation, and the unweighted ensemble.

This result is consistent with the history of methylation clocks. Several landmark clocks were built with penalized regression, not because flexible models were unavailable, but because methylation data are high-dimensional, correlated, and often analyzed in modest sample sizes [2,3,6-8,36]. When a feature panel has already been enriched for age-informative CpGs, the remaining predictive problem may be close to linear over much of the observed age range. A flexible model can still fit nonlinearities, but it can also fit cohort-specific structure that does not travel. In this study, the cost of that flexibility was visible in external mean absolute error.

The negative result is useful precisely because the study did not ask a vague question of whether machine learning can model methylation age. All models in the comparison were machine-learning models in the broad sense. The question was whether additional flexibility improved external performance after the biologically selected panel had already concentrated much of the age signal. The answer was no in these two transfer experiments. That finding supports a practical default: begin with regularized linear models, require external evidence before accepting added complexity, and treat an ensemble as an empirical hypothesis rather than an automatic upgrade.

The result does not imply that nonlinear or deep models are unhelpful for all epigenetic-clock problems. Flexible models may be useful when learning from genome-wide CpG sets, modeling tissue-specific dynamics, incorporating longitudinal outcomes, improving assay efficiency, or integrating additional biological structure [13,21-32]. The claim here is narrower: on the 418-CpG union of established clock probes, under bidirectional external validation between these two GEO cohorts, added complexity did not improve chronological-age prediction. That narrower claim is important because many practical clock applications use preselected CpGs or established panels rather than open-ended feature discovery.

The finding also has implications for how new clock studies should present model comparisons. A complex model can be worth using if it improves external performance, improves calibration, improves interpretability, or captures a biological endpoint that a simpler model misses. It is less compelling when it only improves apparent fit inside a cohort or adds a visually attractive architecture. In this study, the simplest family of models was not merely competitive; it was the external benchmark that other models failed to beat.

The conformal results add a second layer of caution. Primary ElasticNetCV split-conformal intervals achieved empirical coverage 0.898 in the GSE40279 to GSE87571 direction, near nominal coverage 0.90. In the reverse direction, empirical coverage was 0.748. This asymmetry is not a contradiction of conformal theory. The usual split-conformal finite-sample guarantee relies on exchangeability between calibration and test data [33,34]. Cross-cohort transfer can violate that condition through cohort composition, processing differences, methylation distribution shifts, or residual biological differences. The practical lesson is that conformal prediction should be reported with empirical external coverage whenever possible, not simply asserted as calibrated because the method is distribution-free under its assumptions.

The width of the conformal intervals also changes the way individual methylation-age predictions should be communicated. Point errors of 3.79 to 3.95 years for the best linear models may sound precise at the population level. Primary ElasticNetCV individual intervals with mean widths of 15.09 and 11.08 years are a different kind of object. They communicate that individual clock predictions are uncertain even when aggregate correlation is high. For translational use, this is a feature rather than a defect. It makes the uncertainty visible and reduces the temptation to overread a single predicted age.

A practical reporting standard follows from this result. A methylation-clock paper that reports individual predictions should distinguish point accuracy from interval validity. A high Pearson correlation can coexist with wide individual uncertainty, and a conformal method can be correctly implemented while still undercovering under cohort transfer. Reporting nominal coverage alone is therefore incomplete. Reporting empirical coverage by external cohort is more informative and more honest.

The SHAP analysis supports a similarly careful interpretation of feature importance. The dominance of cg16867657 was strong and stable. It ranked first in both directions and had average mean absolute SHAP 7.93, far above the second-ranked CpG at 1.30. This fits the established role of ELOVL2 methylation as an age-associated marker [10,11]. At the same time, the whole-ranking stability was weak. Spearman rank correlation across all 418 CpGs was 0.290, and the top-20 Jaccard index was 0.18, with 6 overlapping CpGs. The phrase "SHAP identified the important CpGs" would therefore be too broad. The more accurate statement is that SHAP identified a highly stable dominant CpG and a less stable attribution tail.

This distinction matters for biological interpretation. Epigenetic clocks often mix causal, consequential, developmental, cell-composition, and technical signals [4,5,20-22,26]. A CpG can improve prediction without being a causal driver of aging, and a model explanation can be stable for prediction without proving mechanism. The present analysis was not designed to infer causality. It was designed to ask whether predictive importance transferred across cohorts. The answer was mixed: one anchor transferred clearly, but broad importance rankings did not.

The age-gap sex analysis also argues for restraint. The revised analysis avoided direct sex leakage by using ElasticNetCV predictors trained without sex. In one direction, the age-adjusted female coefficient was -0.29 years with p = 0.332. In the reverse direction, the age-adjusted female coefficient was -1.27 years with p = 0.000176. A naive reading might focus on the lower p value in the reverse direction. The bidirectional design makes that reading less persuasive. If a downstream association appears in one transfer direction but not the other, it may reflect transfer asymmetry, model calibration, cohort structure, or true biology that is expressed differently across cohorts. This study cannot resolve those alternatives. It can show that downstream age-gap associations are not automatically stable.

This point extends beyond sex as a covariate. Many methylation-clock studies regress age gaps on exposures, diseases, behaviors, or social variables. Those analyses can be valuable, but they depend on the measurement properties of the age gap. If the age gap changes with model class, training cohort, calibration behavior, or transfer direction, then downstream associations may partly reflect measurement instability. The present analysis suggests that downstream age-gap studies should include sensitivity analyses across predictors and validation cohorts whenever possible.

Several strengths follow from the study design. The first is bidirectional external validation. Rather than reporting a single development-to-test split, the analysis trained each cohort against the other. This made it possible to observe that point-prediction performance, conformal coverage, and sex-associated age gaps all changed with transfer direction. The second strength is the fixed feature panel. By restricting the analysis to the 418-CpG union of established clock probes, the study avoided turning the model comparison into a genome-wide feature-search exercise. However, because GSE40279 was the Hannum-clock discovery cohort, this strength applies to the common fixed-panel model-class comparison, not to an historically independent feature-selection claim. The third strength is the layered evaluation: point accuracy, uncertainty calibration, feature attribution stability, paired bootstrap model comparison, and exploratory no-sex age-gap association were all examined from the same external-validation frame.

The fixed-panel design also makes the parsimony result easier to interpret. A genome-wide comparison could be criticized if different algorithms selected different features, because model class and feature discovery would be entangled. Here, every model received the same CpG panel and the same training-test directions. The comparison therefore focused on the mapping from established methylation features to age, which is the practical choice facing many analysts who reuse known clock probes.

The limitations are also clear. First, the analysis used two cohorts. A larger multi-cohort design would better separate cohort-specific effects from general transfer behavior. Second, the feature panel was intentionally restricted and included CpGs originally selected with GSE40279 as the Hannum discovery cohort. That restriction supports the parsimony question but does not test whether genome-wide models, newly selected panels, or newer assay-specific clocks could perform better. Third, the conformal intervals were symmetric absolute-error intervals for ElasticNetCV, with the ensemble retained only as a secondary comparison. Other conformal methods, including locally adaptive or covariate-conditional approaches, might improve efficiency or transfer, but they would require separate validation. Fourth, the age-gap sex analysis was exploratory even after removing sex from the predictor and adjusting for chronological age. Fifth, public GEO data are useful for reproducibility but do not contain every technical or clinical covariate that might explain transfer differences.

Another limitation is that the study did not attempt to harmonize cohorts beyond the preprocessing pipeline. More aggressive harmonization, cell-composition adjustment, or batch correction could change the balance between model classes. Those steps were not added because the analysis was designed as a transparent cross-cohort benchmark using the available GEO-derived files. The resulting design is intentionally conservative: it tests what happens when a clock model is trained in one cohort and applied to another without tailoring the test cohort to the model.

These limitations point to straightforward future work. Multi-cohort validation should become a default expectation for methylation-clock claims that are intended to generalize beyond a development cohort. Conformal uncertainty should be evaluated under explicit cohort-transfer settings, with undercoverage reported rather than hidden. Feature-attribution studies should include rank-stability analyses, not just ranked bar plots. Finally, downstream analyses of age acceleration should be repeated across transfer directions and, where possible, across multiple predictors, because biological interpretation depends on measurement stability.

The main practical conclusion is conservative. In this study, the best external performance came from simple regularized linear modeling. The ensemble did not win, this multilayer-perceptron implementation performed poorly in external transfer, conformal uncertainty was only partly calibrated across cohorts, and SHAP supported one stable CpG anchor rather than a fully stable biological ranking. These are not failures of methylation clocks. They are reminders that strong age signal, high correlation, and sophisticated machine learning do not remove the need for external validation, calibrated uncertainty, and cautious interpretation.

## Conclusion

On a 418-CpG union of established methylation-clock probes, bidirectional external validation showed that regularized linear models matched or outperformed more complex machine-learning models. ElasticNetCV reached mean absolute error 3.95 years when trained on GSE40279 and tested on GSE87571, and 3.79 years in the reverse direction. Paired bootstrap intervals showed no evidence that ElasticNetCV and the plain elastic-net baseline differed in the first direction, but supported ElasticNetCV over more complex models in the reverse direction. Primary ElasticNetCV split-conformal intervals were empirically near nominal in one direction, with coverage 0.898 at nominal coverage 0.90, but undercovered in the reverse direction, with coverage 0.748. SHAP analysis identified cg16867657 as a stable dominant CpG, but broader feature rankings were weakly stable. The study therefore supports a parsimony-first, uncertainty-aware interpretation of epigenetic-clock machine learning on established CpG panels.

## Figure legends

**Figure 1. External ElasticNetCV predicted age versus chronological age.** Scatterplots show ElasticNetCV predicted age against chronological age for train GSE40279, test GSE87571 and train GSE87571, test GSE40279. The diagonal line indicates perfect agreement.

**Figure 2. External-validation mean absolute error by model.** Bars compare plain elastic net, ElasticNetCV, histogram gradient boosting, multilayer perceptron, and the unweighted ensemble in both transfer directions.

**Figure 3. Top CpG SHAP importance values.** Bars show the top 20 CpGs by average mean absolute SHAP value across the two gradient-boosting external-validation directions.

**Figure 4. Split-conformal external interval coverage.** Bars show empirical coverage for nominal 0.90 primary ElasticNetCV split-conformal intervals in each transfer direction.

**Figure 5. External-test age-gap distributions.** Histograms show no-sex ElasticNetCV age gaps, defined as predicted age minus chronological age, for both transfer directions.

## Declarations

### Funding

No specific funding was received for this study.

### Competing interests

The author declares no competing interests.

### Ethics approval

This study used public, de-identified Gene Expression Omnibus data from GSE40279 and GSE87571. No new human participants were recruited, no new biospecimens were collected, and no attempt was made to identify participants. The analysis was therefore a secondary analysis of public de-identified data.

### Data availability

The source methylation datasets are available through the Gene Expression Omnibus under accessions GSE40279 and GSE87571. The analysis outputs used for this manuscript are stored in `results/`, including `external_validation.json`, `conformal_results.json`, `model_comparison_bootstrap.json`, `shap_results.json`, `shap_importance.csv`, `age_gap_results.json`, `age_gap_no_sex_predictions.csv`, and `metrics.json`. The analysis code is available at [repository link redacted for review].

### Code availability

The analysis was generated by `make_clock.py`, with figures generated by `make_figures.py`. The manuscript reports only numbers present in the analysis outputs and `ARTICLE3_methylation_facts.md`.

### Author contributions

The author conceived the study, performed the analysis, interpreted the results, and prepared the manuscript.

### Generative artificial intelligence disclosure

A large language model was used for drafting and code-assistance support. The author retained full responsibility for the data, analysis, interpretation, conclusions, and final manuscript content.

## References

1. Bocklandt S, Lin W, Sehl ME, Sanchez FJ, Sinsheimer JS, Horvath S, Vilain E. Epigenetic predictor of age. PLoS ONE. 2011;6:e14821. https://doi.org/10.1371/journal.pone.0014821

2. Hannum G, Guinney J, Zhao L, Zhang L, Hughes G, Sadda S, et al. Genome-wide methylation profiles reveal quantitative views of human aging rates. Molecular Cell. 2013;49:359-367. https://doi.org/10.1016/j.molcel.2012.10.016

3. Horvath S. DNA methylation age of human tissues and cell types. Genome Biology. 2013;14:R115. https://doi.org/10.1186/gb-2013-14-10-r115

4. Horvath S, Raj K. DNA methylation-based biomarkers and the epigenetic clock theory of ageing. Nature Reviews Genetics. 2018;19:371-384. https://doi.org/10.1038/s41576-018-0004-3

5. Bell CG, Lowe R, Adams PD, Baccarelli AA, Beck S, Bell JT, et al. DNA methylation aging clocks: challenges and recommendations. Genome Biology. 2019;20:249. https://doi.org/10.1186/s13059-019-1824-y

6. Levine ME, Lu AT, Quach A, Chen BH, Assimes TL, Bandinelli S, et al. An epigenetic biomarker of aging for lifespan and healthspan. Aging. 2018;10:573-591. https://doi.org/10.18632/aging.101414

7. Lu AT, Quach A, Wilson JG, Reiner AP, Aviv A, Raj K, et al. DNA methylation GrimAge strongly predicts lifespan and healthspan. Aging. 2019;11:303-327. https://doi.org/10.18632/aging.101684

8. Belsky DW, Caspi A, Corcoran DL, Sugden K, Poulton R, Arseneault L, et al. DunedinPACE, a DNA methylation biomarker of the pace of aging. eLife. 2022;11:e73420. https://doi.org/10.7554/eLife.73420

9. Marioni RE, Shah S, McRae AF, Chen BH, Colicino E, Harris SE, et al. DNA methylation age of blood predicts all-cause mortality in later life. Genome Biology. 2015;16:25. https://doi.org/10.1186/s13059-015-0584-6

10. Garagnani P, Bacalini MG, Pirazzini C, Gori D, Giuliani C, Mari D, et al. Methylation of ELOVL2 gene as a new epigenetic marker of age. Aging Cell. 2012;11:1132-1134. https://doi.org/10.1111/acel.12005

11. El-Shishtawy NM, El Marzouky FM, El-Hagrasy HA. DNA methylation of ELOVL2 gene as an epigenetic marker of age among Egyptian population. Egyptian Journal of Medical Human Genetics. 2024;25. https://doi.org/10.1186/s43042-024-00477-7

12. Tomo Y, Nakaki R. Transfer Elastic Net for developing epigenetic clocks for the Japanese population. Mathematics. 2024;12:2716. https://doi.org/10.3390/math12172716

13. Prosz A, Pipek O, Borcsok J, Palla G, Szallasi Z, Spisak S, Csabai I. Biologically informed deep learning for explainable epigenetic clocks. Scientific Reports. 2024;14. https://doi.org/10.1038/s41598-023-50495-5

14. Mavrommatis C, Belsky DW, Ying K, Moqri M, Campbell A, Richmond A, et al. An unbiased comparison of 14 epigenetic clocks in relation to 174 incident disease outcomes. Nature Communications. 2025;16. https://doi.org/10.1038/s41467-025-66106-y

15. Apsley AT, Ye Q, Caspi A, Chiaro C, Etzel L, Hastings WJ, et al. Cross-tissue comparison of epigenetic aging clocks in humans. Aging Cell. 2025;24. https://doi.org/10.1111/acel.14451

16. Garma LD, Quintela-Fandino M. Applicability of epigenetic age models to next-generation methylation arrays. Genome Medicine. 2024;16. https://doi.org/10.1186/s13073-024-01387-4

17. Lussier AA, Schuurmans IK, Grossbach A, MacIsaac JL, Dever K, Koen N, et al. Technical variability across the 450K, EPICv1, and EPICv2 DNA methylation arrays: lessons learned for clinical and longitudinal studies. Clinical Epigenetics. 2024;16. https://doi.org/10.1186/s13148-024-01761-4

18. Khodasevich D, Gladish N, Daredia S, Bozack AK, Shen H, Nwanaji-Enwerem JC, et al. Influence of race, ethnicity, and sex on the performance of epigenetic predictors of phenotypic traits. Clinical Epigenetics. 2025;17. https://doi.org/10.1186/s13148-025-01864-6

19. Apsley AT, Etzel L, Ye Q, Shalev I. From population science to the clinic? Limits of epigenetic clocks as personal biomarkers. Epigenomics. 2025. https://doi.org/10.1080/17501911.2025.2603880

20. Ying K, Liu H, Tarkhov AE, Sadler MC, Lu AT, Moqri M, et al. Causality-enriched epigenetic age uncouples damage and adaptation. Nature Aging. 2024;4. https://doi.org/10.1038/s43587-023-00557-0

21. Tong H, Dwaraka VB, Chen Q, Luo Q, Lasky-Su J, Smith R, Teschendorff AE. Quantifying the stochastic component of epigenetic aging. Nature Aging. 2024;4. https://doi.org/10.1038/s43587-024-00600-8

22. Tarkhov AE, Lindstrom-Vautrin T, Zhang S, Ying K, Moqri M, Zhang B, et al. Nature of epigenetic aging from a single-cell perspective. Nature Aging. 2024;4. https://doi.org/10.1038/s43587-024-00616-0

23. Tomusiak A, Floro A, Tiwari R, Riley R, Matsui H, Andrews N, et al. Development of an epigenetic clock resistant to changes in immune cell composition. Communications Biology. 2024;7. https://doi.org/10.1038/s42003-024-06609-4

24. Zoller JA, Horvath S. MammalMethylClock R package: software for DNA methylation-based epigenetic clocks in mammals. Bioinformatics. 2024;40. https://doi.org/10.1093/bioinformatics/btae280

25. Shokhirev MN, Torosin NS, Kramer DJ, Johnson AD, Cuellar T. CheekAge: a next-generation buccal epigenetic aging clock associated with lifestyle and health. GeroScience. 2024;46. https://doi.org/10.1007/s11357-024-01094-3

26. Crimmins EM, Klopack ET, Kim JK. Generations of epigenetic clocks and their links to socioeconomic status in the Health and Retirement Study. Epigenomics. 2024;16. https://doi.org/10.1080/17501911.2024.2373682

27. Jain N, Li J, Lin T, Jasmine F, Kibriya MG, Demanelis K, et al. DNA methylation correlates of chronological age in diverse human tissue types. Epigenetics & Chromatin. 2024;17. https://doi.org/10.1186/s13072-024-00546-6

28. Marttila S, Rajic S, Ciantar J, Mak JKL, Junttila I, Kummola L, et al. Biological aging of different blood cell types. GeroScience. 2024. https://doi.org/10.1007/s11357-024-01287-w

29. Zarandooz S, Raffington L. Applying blood-derived epigenetic algorithms to saliva: cross-tissue similarity of DNA-methylation indices of aging, physiology, and cognition. Clinical Epigenetics. 2025;17. https://doi.org/10.1186/s13148-025-01868-2

30. Griffin P, Kane AE, Trapp A, Li J, Arnold M, Poganik JR, et al. TIME-seq reduces time and cost of DNA methylation measurement for epigenetic clock construction. Nature Aging. 2024;4. https://doi.org/10.1038/s43587-023-00555-2

31. Hu C, Li Y, Li L, Zhang N, Zheng X. BS-clock, advancing epigenetic age prediction with high-resolution DNA methylation bisulfite sequencing data. Bioinformatics. 2024;40. https://doi.org/10.1093/bioinformatics/btae656

32. Fuentealba M, Rouch L, Guyonnet S, Lemaitre JM, de Souto Barreto P, Vellas B, et al. A blood-based epigenetic clock for intrinsic capacity predicts mortality and is associated with clinical, immunological and lifestyle factors. Nature Aging. 2025;5. https://doi.org/10.1038/s43587-025-00883-5

33. Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction. Foundations and Trends in Machine Learning. 2023;16:494-591. https://doi.org/10.1561/2200000101

34. Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a Random World. 2nd ed. Springer; 2022. https://doi.org/10.1007/978-3-031-06649-8

35. Lundberg SM, Erion G, Chen H, DeGrave A, Prutkin JM, Nair B, et al. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence. 2020;2:56-67. https://doi.org/10.1038/s42256-019-0138-9

36. Zou H, Hastie T. Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society Series B: Statistical Methodology. 2005;67:301-320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

37. Friedman JH. Greedy function approximation: a gradient boosting machine. The Annals of Statistics. 2001;29:1189-1232. https://doi.org/10.1214/aos/1013203451
