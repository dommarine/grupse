import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import curve_fit

def fitting_func(x, a0, b):
    return a0 *np.exp(b * x)

def growth_func(t, q, P_0, X_0, mu, k=0, A=1, B=0, P_0g=0, q_g=0):
    return A*(P_0 * np.exp(-k * t) + ((q * X_0) / (k + mu)) * (np.exp(mu * t) - np.exp(k * t))) + B*((P_0 + ((((q * X_0) / mu) - ((q_g * X_0 / (mu + k)) * (k / mu))) * (np.exp(mu * t) - 1)) + (P_0g + ((q_g * X_0) / (mu + k))) * (1 - np.exp(-k * t))))

def calculate_aic(n, mse, num_params):
    return 2 * num_params + n * np.log(mse)

@dataclass
class GeneralCellGrowthData:
    df: pd.DataFrame
    lower_bound: int
    upper_bound: int
    p0: tuple[float, float]

@dataclass
class CellGrowthFitting:
    xdata: pd.Series
    ydata: pd.Series
    xdata_filtered_inliers: pd.Series
    ydata_filtered_inliers: pd.Series
    xdata_filtered_outliers: pd.Series
    ydata_filtered_outliers: pd.Series
    x_fit: np.ndarray
    popt: np.ndarray
    funcdata: np.ndarray

@dataclass
class OptimumParameters:
    a0: float
    b: float
    confidence_interval_a0: float
    confidence_interval_b: float
    r_squared: float




def calculate_cell_growth_fitting(
        cell_growth_data: GeneralCellGrowthData,
        use_ransac_regression: bool,
        use_extrapolation: bool):

    xdata = cell_growth_data.df[cell_growth_data.df.columns[0]]
    ydata = cell_growth_data.df[cell_growth_data.df.columns[1]]

    xdata_filtered = xdata[(cell_growth_data.lower_bound <= xdata) & (xdata <= cell_growth_data.upper_bound)]
    ydata_filtered = ydata[(cell_growth_data.lower_bound <= xdata) & (xdata <= cell_growth_data.upper_bound)]

    if use_ransac_regression:
        ransac = RANSACRegressor()
        ransac.fit(xdata_filtered.values.reshape(-1, 1), ydata_filtered)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        xdata_filtered_inliers = xdata_filtered[inlier_mask]
        ydata_filtered_inliers = ydata_filtered[inlier_mask]
        xdata_filtered_outliers = xdata_filtered[outlier_mask]
        ydata_filtered_outliers = ydata_filtered[outlier_mask]
    else:
        xdata_filtered_inliers = xdata_filtered
        ydata_filtered_inliers = ydata_filtered
        xdata_filtered_outliers = pd.Series([], dtype=xdata_filtered.dtype)
        ydata_filtered_outliers = pd.Series([], dtype=ydata_filtered.dtype)

    popt, _ = curve_fit(fitting_func, xdata_filtered_inliers, ydata_filtered_inliers, p0=cell_growth_data.p0)
    funcdata = fitting_func(xdata_filtered_inliers, *popt)
    x_fit = np.linspace(min(xdata_filtered_inliers), cell_growth_data.upper_bound + (20 if use_extrapolation else 0), 100)

    return CellGrowthFitting(
        xdata=xdata,
        ydata=ydata,
        xdata_filtered_inliers=xdata_filtered_inliers,
        ydata_filtered_inliers=ydata_filtered_inliers,
        xdata_filtered_outliers=xdata_filtered_outliers,
        ydata_filtered_outliers=ydata_filtered_outliers,
        x_fit=x_fit,
        popt=popt,
        funcdata=funcdata
    )

def calculate_cell_growth_optimum_parameters(cell_growth_fitting: CellGrowthFitting):
    residuals = cell_growth_fitting.ydata_filtered_inliers - cell_growth_fitting.funcdata
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((cell_growth_fitting.ydata_filtered_inliers - np.mean(cell_growth_fitting.ydata_filtered_inliers)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    _, pcov = curve_fit(fitting_func, cell_growth_fitting.xdata_filtered_inliers, cell_growth_fitting.ydata_filtered_inliers, p0=cell_growth_fitting.popt)
    std_devs = np.sqrt(np.diag(pcov))
    conf_int = 1.96 * std_devs

    return OptimumParameters(
        a0=cell_growth_fitting.popt[0],
        b=cell_growth_fitting.popt[1],
        confidence_interval_a0=conf_int[0],
        confidence_interval_b=conf_int[1],
        r_squared=r_squared
    )


def create_cell_growth_plot(fitting: CellGrowthFitting, use_log_scale: bool = False):
    fig, ax = plt.subplots()

    ax.plot(fitting.x_fit, fitting_func(fitting.x_fit, *fitting.popt), color='#D95F02', label='Fit')
    ax.scatter(fitting.xdata, fitting.ydata, label='Data', color='#1B9E77')
    ax.scatter(fitting.xdata_filtered_inliers, fitting.ydata_filtered_inliers, label='Data used for fit', color='#7570B3')

    if not fitting.xdata.empty:
        ax.scatter(fitting.xdata_filtered_outliers, fitting.ydata_filtered_outliers, label='Outliers', color='#A2AAAD')

    ax.set_xlabel('Time [h]')
    ax.set_ylabel('viable cells [10⁶ viable cells·ml⁻¹]')
    ax.set_title(f"Time vs. Cells{' (log scale)' if use_log_scale else ''}")
    
    if use_log_scale:
        ax.set_yscale('log')
    
    ax.legend()

    return fig



