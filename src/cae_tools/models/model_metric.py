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

    def accumulate(self, actual: np.ndarray, estimates: np.ndarray):
        if actual.shape != estimates.shape:
            raise ValueError("The shapes of 'actual' and 'estimates' must match.")

        # accumulate the actual and estimated data

        self.actuals.append(actual)
        self.estimates.append(estimates)

    def get_metrics(self):
        # input checks
        if not self.actuals or not self.estimates:
            raise ValueError("No data accumulated to calculate metrics.")

        # concatenate the accumulated data
        all_actuals = np.concatenate(self.actuals)
        all_estimates = np.concatenate(self.estimates)

        #  MSE
        mse = np.mean((all_actuals - all_estimates) ** 2)

        #  RMSE
        rmse = np.sqrt(mse)

        #  MAE
        mae = np.mean(np.abs(all_actuals - all_estimates))

        # Pearson correlation (mean across all instances)
        pearson_correlations = []
        for actual, estimate in zip(self.actuals, self.estimates):
            # flatten each instance to 1D
            actual_flat = actual.flatten()
            estimate_flat = estimate.flatten()

            # calculate Pearson correlation
            correlation, _ = pearsonr(actual_flat, estimate_flat)
            pearson_correlations.append(correlation)

        # mean Pearson correlation across all instances
        mean_pearson_correlation = np.mean(pearson_correlations)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mean_pearson_correlation": mean_pearson_correlation
        }
