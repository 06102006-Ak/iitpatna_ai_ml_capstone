# Analytics EDA Report

Cleaned shape: **889 rows × 14 columns**. Overall survival rate: **38.25%**.

## Missing-value profile and threshold decisions

| Column | Missing % | Decision |
|---|---:|---|
| `deck` | 77.22% | drop column — >30% missing; unreliable direct imputation and not needed downstream |
| `age` | 19.87% | median imputation — 5%–30% missing |
| `embarked` | 0.22% | drop rows — under 5% missing |
| `embark_town` | 0.22% | drop rows — under 5% missing |

The under-5% fields were handled by row deletion, the 5%–30% numeric field(s) were median-imputed, and the very-high-missing `deck` field was dropped rather than imputed because it is structurally sparse and not required by the downstream model.

## Univariate analysis

| Variable | IQR lower fence | IQR upper fence | Outliers |
|---|---:|---:|---:|
| `age` | 2.500 | 54.500 | 65 |
| `fare` | -26.761 | 65.656 | 114 |

Fare mean = **32.097**, median = **14.454**, mode = **8.050**. Because the mean is above the median, which is above the mode, the fare distribution is **right-skewed**, consistent with a long upper tail of higher-priced tickets.

## Bivariate analysis

### Survival by sex
|        |   survival_rate |
|:-------|----------------:|
| female |          0.7404 |
| male   |          0.1889 |

### Survival by passenger class
|    |   survival_rate |
|---:|----------------:|
|  1 |          0.6262 |
|  2 |          0.4728 |
|  3 |          0.2424 |

### Survival by sex and passenger class
| Sex / class | Survival rate |
|---|---:|
| female / 1 | 96.74% |
| female / 2 | 92.11% |
| female / 3 | 50.00% |
| male / 1 | 36.89% |
| male / 2 | 15.74% |
| male / 3 | 13.54% |

### Correlation matrix

The required matrix uses exactly `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`; the boolean `adult_male` and `alone` flags are excluded.

The two strongest absolute off-diagonal correlations are **pclass ↔ fare (r=-0.548)** and **sibsp ↔ parch (r=0.415)**. These values indicate that the first pair moves most closely together linearly, while the second pair shows the next-largest magnitude relationship; neither correlation by itself establishes causality.

## Multivariate data story

**Chart 1 — survival by sex and class.** The grouped bars show that survival varies jointly by sex and passenger class rather than by either feature in isolation. Within each class, the two sex groups have visibly different survival rates, while higher classes also show different outcomes.

**Chart 2 — age by survival and sex.** The box plots compare age distributions simultaneously across survival outcomes and sex. The overlap means age alone does not explain survival, but age composition differs across these groups and should be retained as a contextual feature rather than treated as a deterministic rule.

**Chart 3 — fare vs age by survival and class.** The scatter plot shows how fare, age, class, and survival interact, with higher fares concentrated in particular class groupings. This supports using multiple variables together instead of relying on a single univariate threshold.

**Chart 4 — embarkation point and sex.** The heatmap shows that survival rates differ across embarkation locations within sex categories. This is an example of an interaction pattern that would be obscured by looking only at a global embarkation average.

## Exploratory standardization sanity check

| Column | Before mean | Before std | Z-score mean | Z-score std |
|---|---:|---:|---:|---:|
| `age` | 29.3152 | 12.9849 | 0.000000 | 1.000000 |
| `fare` | 32.0967 | 49.6975 | 0.000000 | 1.000000 |

The standardized columns have means approximately zero and sample standard deviations approximately one, confirming the transformation. This exploratory check is separate from the modeling pipeline's own train-only scaling.

## Saved charts

`age_hist.png`, `age_box.png`, `fare_hist.png`, `fare_box.png`, `correlation_heatmap.png`, `multivariate_survival_sex_class.png`, `multivariate_age_survival_sex.png`, `multivariate_fare_age_survival_class.png`, and `multivariate_embarked_sex_heatmap.png` are supporting artifacts; the interpretations above are the required textual evidence.
