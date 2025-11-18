# SCALER CONFIGURATION

## Important: Add Your Scaler File

If your model uses preprocessing (scaling, normalization, or encoding), please add your `scaler.pkl` file to this directory.

### How to create and save a scaler:

```python
import joblib
from sklearn.preprocessing import StandardScaler  # or MinMaxScaler, RobustScaler, etc.

# After fitting your scaler during training:
joblib.dump(scaler, 'scaler.pkl')
```

### Supported Preprocessing Objects:

- StandardScaler
- MinMaxScaler
- RobustScaler
- LabelEncoder
- OneHotEncoder
- Any custom preprocessing pipeline

### If you don't use a scaler:

The app will work fine without it! It will simply skip the preprocessing step and use raw input values for predictions.

---

**Note:** Make sure the scaler was fitted on the same features and in the same order as your training data.
