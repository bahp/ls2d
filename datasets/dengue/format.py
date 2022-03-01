# Libraries
import pandas as pd
import numpy as np
import pickle

# Path
path = './combined_tidy_v0.0.10.csv'

# Read data ad model
data = pd.read_csv(path, parse_dates=['date'])

# ---------------------------------------------------
# Data pre-processing and adding columns
# ---------------------------------------------------
""".. note: Instead of just keeping the rows that have
            the full set of features, it would also be
            possible to input this missing value using
            the IterativeImputer.

   .. note: In the projections, should I scale the data
            with the whole dataset instead of using the
            aggregated data? (data_f)      
"""
# Ugly 06dx numbers
data.study_no = data.study_no \
    .str.lower() \
    .str.replace('-06dx', '-')

# Add some conditions
data['ast>1000'] = data.ast.fillna(0) >= 1000
data['alt>1000'] = data.alt.fillna(0) >= 1000
data['liver>2'] = data.liver_palpation_size.fillna(0) > 2

"""
# .. note: Should we include data.oedema? Also, it
#          would be possible to move the combination
#          of pleural_effusion and oedema_pulmonary
#          so it appears before fluid_accumulation
# Fluid accumulation
data['fluid_accumulation'] = \
    data.ascites | \
    data.effusion | \
    data.pleural_effusion | \
    data.oedema_pulmonary
"""

# Liver abnormal
data['liver_abnormal'] = \
    data.liver_acute | \
    data.liver_involved | \
    data.liver_failure | \
    data.liver_severe | \
    data.jaundice | \
    data['liver>2']

# Kidney abnormal
data['kidney_abnormal'] = \
    data.skidney

# Pleural effusion
data['pleural_effusion'] = \
    data.pleural_effusion | \
    data.effusion

# Oedema pulmonary
data['oedema_pulmonary'] = \
    data.oedema_pulmonary | \
    data.oedema

# Create features
data['severe_leak'] = \
    data.ascites | \
    data.overload | \
    data.oedema_pulmonary | \
    data.respiratory_distress | \
    data.oedema | \
    data.pleural_effusion | \
    data.effusion

# Bleeding
data['severe_bleed'] = \
    data.bleeding_gi | \
    data.bleeding_urine

# Organ impairment
data['severe_organ'] = \
    data.cns_abnormal | \
    data.neurology.astype(bool) | \
    (data.ast.fillna(0) >= 1000) | \
    (data.alt.fillna(0) >= 1000) | \
    data.liver_abnormal | \
    data.kidney_abnormal

# Category: severe
data['severe'] = \
    data.severe_leak | \
    data.severe_bleed | \
    data.severe_organ | \
    data.shock

# Category: warning WHO
data['warning'] = \
    data.abdominal_pain | \
    data.abdominal_tenderness | \
    data.vomiting | \
    data.ascites | \
    data.pleural_effusion | \
    data.bleeding_mucosal | \
    data.restlessness | \
    data.lethargy | \
    (data.liver_palpation_size.fillna(0) > 2)

# Propagate for whole stay
data.severe = data \
    .groupby('study_no') \
    .severe.transform('max')

data.warning = data \
    .groupby('study_no') \
    .warning.transform('max')

# Add mild
data['mild'] = ~(data.severe | data.warning)


# ----------------
# Filter
# ----------------

def IQR(values, q1=0.25, q3=0.75, replace=np.nan):
    """Filter outliers

    .. note: Where cond is True, keep the original
             value. Where False, replace with
             corresponding value from other.
    """
    # Calculate quantiles and IQR
    Q1 = values.quantile(q1)
    Q3 = values.quantile(q3)
    IQR = Q3 - Q1

    # Condition
    condition = (values > (Q1 - 1.5 * IQR)) | (values < (Q3 + 1.5 * IQR))

    # Return
    return values.where(condition, replace)


# Filter
data = data[data.age.between(0.0, 18.0)]
data = data[data.plt < 50000]
data.plt = IQR(data.plt)

# Show
print("\nData:")
print(data)

# Save
data.to_csv('data.csv', index=False)
