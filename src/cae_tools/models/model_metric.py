#    Copyright (C) 2023  National Centre for Earth Observation (NCEO)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.stats import pearsonr

class ModelMetric:

    def __init__(self):
        self.actuals = []
        self.estimates = []

    def accumulate(self, actual: np.ndarray, estimates: np.ndarray, mask: np.ndarray):
        """
        Accumulate actual and estimated data for a single instance.
        Both actual and estimates should have the same shape.
        The mask should have the same spatial shape (or broadcastable shape) and be binary.
        Only the values where mask == 1 are accumulated.
        """
        if actual.shape != estimates.shape:
            raise ValueError("The shapes of 'actual' and 'estimates' must match.")
        
        # flatten the arrays
        actual_flat = actual.flatten()
        estimates_flat = estimates.flatten()
        mask_flat = mask.flatten().astype(bool)  # Convert mask to boolean
        
        # take only the elements where mask is True
        valid_actual = actual_flat[mask_flat]
        valid_estimates = estimates_flat[mask_flat]
        
        self.actuals.append(valid_actual)
        self.estimates.append(valid_estimates)

    def get_metrics(self):
        if not self.actuals or not self.estimates:
            raise ValueError("No data accumulated to calculate metrics.")

        all_actuals = np.concatenate(self.actuals)
        all_estimates = np.concatenate(self.estimates)

        mse = np.mean((all_actuals - all_estimates) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_actuals - all_estimates))

        pearson_correlations = []
        for actual, estimate in zip(self.actuals, self.estimates):
            if actual.size == 0 or estimate.size == 0:
                continue
            correlation, _ = pearsonr(actual, estimate)
            pearson_correlations.append(correlation)
        mean_pearson_correlation = np.mean(pearson_correlations) if pearson_correlations else 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mean_pearson_correlation": mean_pearson_correlation
        }

