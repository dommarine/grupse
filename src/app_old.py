import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from shiny import App, render, reactive, req, ui
from sklearn.metrics import r2_score
from sklearn.linear_model import RANSACRegressor
import itertools
import io

# Define function to fit
def fitting_func(x, a0, b):
    return a0 * np.exp(b * x)

# Define the new growth function for metabolites
def growth_func(t, q, P_0, X_0, mu, k=0, A=1, B=0, P_0g=0, q_g=0):
    return A*(P_0 * np.exp(-k * t) + ((q * X_0) / (k + mu)) * (np.exp(mu * t) - np.exp(k * t))) + B*((P_0 + ((((q * X_0) / mu) - ((q_g * X_0 / (mu + k)) * (k / mu))) * (np.exp(mu * t) - 1)) + (P_0g + ((q_g * X_0) / (mu + k))) * (1 - np.exp(-k * t))))

p0 = (1, 0.1)
extension = 20

def calculate_standard_fitting(lower_bound, upper_bound, data, p0, robust=False, use_extension=True):
    xdata = data[data.columns[0]]
    ydata = data[data.columns[1]]
    xdata_filtered = xdata[(lower_bound <= xdata) & (xdata <= upper_bound)]
    ydata_filtered = ydata[(lower_bound <= xdata) & (upper_bound >= xdata)]

    if robust:
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

    popt, _ = curve_fit(fitting_func, xdata_filtered_inliers, ydata_filtered_inliers, p0=p0)
    funcdata = fitting_func(xdata_filtered_inliers, *popt)
    x_fit = np.linspace(min(xdata_filtered_inliers), upper_bound + (extension if use_extension else 0), 100)

    return xdata, ydata, xdata_filtered_inliers, ydata_filtered_inliers, xdata_filtered_outliers, ydata_filtered_outliers, x_fit, popt, funcdata

def plot_cell_graph_normal_scale_standard_fitting(raw_data, prepared_data, use_extension):
    xdata, ydata, xdata_filtered_inliers, ydata_filtered_inliers, xdata_filtered_outliers, ydata_filtered_outliers, x_fit, popt, _ = prepared_data
    plt.plot(x_fit, fitting_func(x_fit, *popt), color='#D95F02', label='Fit')
    plt.scatter(xdata, ydata, label='Data', color='#1B9E77')
    plt.scatter(xdata_filtered_inliers, ydata_filtered_inliers, label='Data used for fit', color='#7570B3')
    if not xdata_filtered_outliers.empty:
        plt.scatter(xdata_filtered_outliers, ydata_filtered_outliers, label='Outliers', color='#A2AAAD')
    plt.xlabel('Time [h]')
    plt.ylabel('viable cells [10⁶ viable cells·ml⁻¹]')
    plt.title('Time vs. Cells')
    plt.legend()
    return plt

def plot_cell_graph_log_scale_standard_fitting(raw_data, prepared_data, use_extension):
    xdata, ydata, xdata_filtered_inliers, ydata_filtered_inliers, xdata_filtered_outliers, ydata_filtered_outliers, x_fit, popt, _ = prepared_data
    plt.xlabel('Time [h]')
    plt.ylabel('Viable Cells [10⁶ Viable Cells·ml⁻¹]')
    plt.yscale('log')
    plt.plot(x_fit, fitting_func(x_fit, *popt), color='#D95F02', label='Fit')
    plt.scatter(xdata, ydata, label='Data', color='#1B9E77')
    plt.scatter(xdata_filtered_inliers, ydata_filtered_inliers, label='Data used for fit', color='#7570B3')
    if not xdata_filtered_outliers.empty:
        plt.scatter(xdata_filtered_outliers, ydata_filtered_outliers, label='Outliers', color='#A2AAAD')
    plt.title('Time vs. Cells (log scale)')
    plt.legend()
    return plt

def calculate_aic(n, mse, num_params):
    return 2 * num_params + n * np.log(mse)

# Define the app UI
app_ui = ui.page_fluid(
    ui.panel_title(ui.HTML('<div style="text-align: center; font-size: 30px;">'
                           '<span style="color: #4C84F3;">G</span>'
                           '<span style="color: #AF6A41;">r</span>'
                           '<span style="color: #EAC30B;">U</span>'
                           '<span style="color: #4C84F3;">p</span>'
                           '<span style="color: #7A9A4E;">S</span>'
                           '<span style="color: #AF6A41;">e</span>'
                           '</div>')),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file("uploaded_csv", "Upload CSV File (cell-growth data)", accept=[".csv"], multiple=False),
            ui.input_numeric("lwr_bound", "Lower bound for fit", 0),
            ui.input_numeric("upr_bound", "Upper bound for fit", 60),
            ui.input_checkbox("use_extension", "Extrapolate", value=True),
            ui.input_file("file", "Upload CSV file (metabolite data)", accept=[".csv"]),
            ui.output_ui("checkboxes_ui"),
            ui.output_text_verbatim("print_fitting_param"),
            ui.input_checkbox("robust_fitting", "Use RANSAC-regression for cell growth", False),
            ui.input_checkbox("robust_fitting_metabolites", "Use RANSAC-regression for metabolites", False),
            ui.input_checkbox("fit_glutamine", "Fit decaying metabolite", False),
            ui.output_ui("glutamine_ui"),
            ui.input_checkbox("fit_ammonia", "Fit decay-product metabolite", False),
            ui.output_ui("ammonia_ui"),
            ui.input_checkbox("convert_rate", "Convert metabolite rates to [mmol·gDW⁻¹h⁻¹]", False),
            ui.input_numeric("dry_mass_per_cell", "Dry mass per cell [pg]", value=250, min=0.001),
            ui.output_text_verbatim("dynamic_params_output")
        ),
        ui.panel_main(
            ui.output_plot("plot_cell_graph_normal_scale"),
            ui.output_ui("plot_cell_graph_normal_scale_download_conditional"),
            ui.output_plot("plot_cell_graph_log_scale"),
            ui.output_ui("plot_cell_graph_log_scale_download_conditional"),
            ui.output_text("text_output"),
            ui.output_plot("plot_output"),
            ui.output_ui("plot_output_conditional"),
        )
    )
)

# Event Handling
def server(input, output, session):
    @reactive.Calc
    def raw_dataframe():
        csv_file = req(input.uploaded_csv())
        return pd.read_csv(csv_file[0]["datapath"])

    @reactive.Calc
    def prepared_data():
        csv_df = raw_dataframe()
        lower_bound, upper_bound = input.lwr_bound(), input.upr_bound()
        p0 = (0.1, 0.12)
        robust = input.robust_fitting()
        use_extension = input.use_extension()
        return calculate_standard_fitting(lower_bound, upper_bound, csv_df, p0, robust, use_extension)

    @reactive.Calc
    def opt_params():
        _, _, xdata_filtered_inliers, ydata_filtered_inliers, _, _, _, popt, funcdata = prepared_data()
        residuals = ydata_filtered_inliers - funcdata
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ydata_filtered_inliers - np.mean(ydata_filtered_inliers)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        _, pcov = curve_fit(fitting_func, xdata_filtered_inliers, ydata_filtered_inliers, p0=popt)
        std_devs = np.sqrt(np.diag(pcov))
        conf_int = 1.96 * std_devs
        return popt[0], popt[1], conf_int[0], conf_int[1], r_squared

    @output
    @render.plot
    def plot_cell_graph_normal_scale():
        use_extension = input.use_extension()
        return plot_cell_graph_normal_scale_standard_fitting(raw_dataframe(), prepared_data(), use_extension).legend()
    
    @output
    @render.ui
    def plot_cell_graph_normal_scale_download_conditional():
        if raw_dataframe() is not None:
            return ui.download_button("plot_cell_graph_normal_scale_download", "Download Normal Scale Graph")
        return None

    @output
    @render.download()
    def plot_cell_graph_normal_scale_download():
        fig, _ = plt.subplots()
        use_extension = input.use_extension()
        plot_cell_graph_normal_scale_standard_fitting(raw_dataframe(), prepared_data(), use_extension)
        fig.savefig("/tmp/cell_graph_normal_scale.png")
        return "/tmp/cell_graph_normal_scale.png"

    @output
    @render.plot
    def plot_cell_graph_log_scale():
        use_extension = input.use_extension()
        return plot_cell_graph_log_scale_standard_fitting(raw_dataframe(), prepared_data(), use_extension).legend()
    
    @output
    @render.ui
    def plot_cell_graph_log_scale_download_conditional():
        if raw_dataframe() is not None:
            return ui.download_button("plot_cell_graph_log_scale_download", "Download Log Scale Graph")
        return None
    
    @output
    @render.download()
    def plot_cell_graph_log_scale_download():
        fig, _ = plt.subplots()
        use_extension = input.use_extension()
        plot_cell_graph_log_scale_standard_fitting(raw_dataframe(), prepared_data(), use_extension)
        fig.savefig("/tmp/cell_graph_log_scale.png")
        return "/tmp/cell_graph_log_scale.png"

    @output
    @render.text
    def print_fitting_param():
        a0, b, ci_a0, ci_b, r_squared = opt_params()
        return "Cell growth parameters ± error\n" \
               f"μ ={b:.9f}±{ci_b:.9f} h⁻¹ \n" \
               f"X₀={a0:.9f}±{ci_a0:.9f} 10⁶cells·mL⁻¹\n" \
               f"R²={r_squared:.9f}"

    @reactive.Calc
    def df():
        file_info = input.file()
        if file_info is None:
            return None
        df = pd.read_csv(file_info[0]["datapath"])
        return df

    @output
    @render.ui
    def checkboxes_ui():
        data = df()
        if data is None:
            return None
        columns = list(data.columns)
        checkboxes = [ui.input_checkbox(f"chk_{col}", col, True) for col in columns[1:]]
        return ui.TagList(*checkboxes)

    @reactive.Calc
    def selected_metabolites():
        data = df()
        if data is None:
            return []
        columns = list(data.columns)
        selected = [col for col in columns[1:] if getattr(input, f"chk_{col}")()]
        return selected

    @output
    @render.ui
    def glutamine_ui():
        selected = input.fit_glutamine()
        if not selected:
            return None
        data = df()
        if data is None:
            return None
        columns = list(data.columns)
        return ui.TagList(
            ui.input_select("glutamine_column", "Select Column", columns[1:]),
            ui.input_numeric("glutamine_k_value", "Enter decay constant k in [h⁻¹]", value=0.0025)
        )

    @output
    @render.ui
    def ammonia_ui():
        selected = input.fit_ammonia()
        if not selected:
            return None
        data = df()
        if data is None:
            return None
        columns = list(data.columns)
        return ui.TagList(
        ui.input_select("ammonia_column", "Select Column", columns[2:]),
    )
    dynamic_output_text = reactive.Value("")

    @output
    @render.text
    def dynamic_params_output():
        return dynamic_output_text.get()

    @output
    @output
    @render.ui
    def plot_output_conditional():
        if df() is not None:
            return ui.download_button("plot_output_download", "Download Metabolite Graph")
        return None

    @output
    @render.download(filename="metabolite_plot.png")
    def plot_output_download():
        if fig is None:
            return None
        
        buf = io.BytesIO()
        fig = plot_output()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        return buf
    
    @reactive.Calc
    def conversion_factor():
        if input.convert_rate():
            return 1 / (input.dry_mass_per_cell() * 1e-3)
        else:
            return 1

    @output
    @render.plot
    def plot_output():
        global I, N
        data = df()
        if data is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        column_list = list(data.columns)
        selected_columns = selected_metabolites()
        dataMet = data.fillna(-1).values.astype(float)
        xTime = dataMet[:, 0]
        lower_bound = input.lwr_bound()
        upper_bound = input.upr_bound()
        use_extension = input.use_extension()
        
        dynamic_outputs = []
        
        # Define the color cycle for data points used for fitting
        fit_colors = ['#7570B3', '#EFCA08', '#90D7FF', '#F49FBC', '#66002C', '#E39EC1']
        color_cycle = itertools.cycle(fit_colors)

        for column in selected_columns:
            i = column_list.index(column)
            df_met = pd.DataFrame({'xTime': xTime, f'yMet{i}': dataMet[:, i]})
            df_met_full = df_met[df_met[f'yMet{i}'] != -1]
            df_met_fit = df_met_full[(df_met_full['xTime'] >= lower_bound) & (df_met_full['xTime'] <= upper_bound)]
            
            fit_color = next(color_cycle)
            
            ax.scatter(df_met_full['xTime'], df_met_full[f'yMet{i}'], label=column, alpha=0.5, color='#1B9E77')
            ax.scatter(df_met_fit['xTime'], df_met_fit[f'yMet{i}'], color=fit_color, label=f'{column} (fit data)', alpha=0.5)
            
            x_fit = df_met_fit['xTime']
            y_fit = df_met_fit[f'yMet{i}']
            
            try:
                # Use the fitted parameters from cell growth fitting
                a0, b, _, _, _ = opt_params()
                if input.robust_fitting_metabolites():
                    ransac = RANSACRegressor()
                    ransac.fit(x_fit.values.reshape(-1, 1), y_fit)
                    inlier_mask = ransac.inlier_mask_
                    outlier_mask = np.logical_not(inlier_mask)
                    x_fit = x_fit[inlier_mask]
                    y_fit = y_fit[inlier_mask]
                    ax.scatter(df_met_fit['xTime'][outlier_mask], df_met_fit[f'yMet{i}'][outlier_mask], color='#A2AAAD', label=f'{column} (outliers)', alpha=0.5)

                # Determine the k value
                if input.fit_glutamine() and column == input.glutamine_column():
                    k_value = input.glutamine_k_value()
                elif input.fit_ammonia() and column == input.ammonia_column():
                    k_value = input.glutamine_k_value()  # Using the same k value for ammonia as for glutamine
                else:
                    k_value = 0  # Default k value for other metabolites

                A, B, q_g, P_0g = 1, 0, 0, 0
                if input.fit_ammonia() and column == input.ammonia_column():
                    q_g = I
                    P_0g = N
                    A, B = 0, 1
                
                # Perform fitting without conversion factor
                popt, pcov = curve_fit(lambda t, q, P_0: growth_func(t, q, P_0, a0, b, k=k_value, A=A, B=B, q_g=q_g, P_0g=P_0g), x_fit, y_fit, p0=p0, bounds=(-200, 200))
                y_pred = growth_func(x_fit, *popt, a0, b, k=k_value, A=A, B=B, q_g=q_g, P_0g=P_0g)
                r_squared = r2_score(y_fit, y_pred)
                perr = np.sqrt(np.diag(pcov))
                confidence_intervals = 1.96 * perr
                x_fit_all = np.linspace(min(df_met_full['xTime']), max(df_met_full['xTime']) + (extension if use_extension else 0), 100)
                x_fit = x_fit_all[(x_fit_all >= lower_bound) & (x_fit_all <= upper_bound + (extension if use_extension else 0))]
                ax.plot(x_fit, growth_func(x_fit, *popt, a0, b, k=k_value, A=A, B=B, q_g=q_g, P_0g=P_0g), label=f'fit {column}', color='#D95F02')
                
                # Apply conversion factor only for display
                q = popt[0] / conversion_factor()
                q_ci = confidence_intervals[0] / conversion_factor()
                P_0 = popt[1]
                P_0_ci = confidence_intervals[1]
                
                # Calculate AIC
                residuals = y_fit - y_pred
                mse = np.mean(residuals**2)
                num_params = len(popt)
                aic = calculate_aic(len(y_fit), mse, num_params)
                
                # Save the optimization results for glutamine
                if input.fit_glutamine() and column == input.glutamine_column():
                    I, N = popt[0], P_0
                
                q_unit = "mmol·gDW⁻¹h⁻¹" if input.convert_rate() else "mmol·10⁶cells⁻¹·h⁻¹"
                dynamic_outputs.append(
                    f"\n{column}: Optimal parameter values ± error\n"
                    f"q   = {q:.9f} ± {q_ci:.9f} {q_unit}\n"
                    f"P₀  = {P_0:.9f} ± {P_0_ci:.9f} mmol·L⁻¹\n"
                    f"R²  = {r_squared:.9f}\n"
                    f"AIC = {aic:.9f}"
                )

            except Exception as e:
                dynamic_outputs.append(f"\nCould not fit the data for {column}: {e}")
        
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Metabolite Concentration [mmol·L⁻¹]')
        ax.set_title('Time vs. Metabolite Concentration')
        ax.set_ylim(bottom=0)
        ax.legend()
        dynamic_output_text.set("\n".join(dynamic_outputs))
        
        return fig

app = App(app_ui, server)