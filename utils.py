import numpy as np

def exponential_transformation(column, bounds, k=1):
    
    
    lower_bound, upper_bound = bounds[0], bounds[1]
    
    
    transformed = np.where(
        (column >= lower_bound) & (column <= upper_bound),  # Condition for being in range
        1,  # Set to 1 if within range
        np.where(
            column < lower_bound,
             np.exp(k * (lower_bound - column)),
             np.exp(k * (column - upper_bound)) 
    ))
     # Apply exponential function otherwise
    return transformed


def exponential_transformation_gender(row, column,gender_col, bounds, k=1):
    
    if row[gender_col] == 1:
        #male
        lower_bound, upper_bound = bounds["male"][0], bounds["male"][1]
    elif row[gender_col] == 0:
        lower_bound, upper_bound = bounds["female"][0], bounds["female"][1]
    else:
        return np.nan  # Unknown gender handling
    
    value = row[column]
    
    # Apply exponential transformation with scaling factor k
    if lower_bound <= value <= upper_bound:
        return 1.0  # Optimal range
    elif value < lower_bound:
        return np.exp(k * (lower_bound - value))  # Exponential penalty below range
    else:
        return np.exp(k * (value - upper_bound))  # Exponential penalty above range