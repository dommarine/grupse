import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

from shiny import App, render, req, reactive, ui
from sklearn.metrics import r2_score
from utils import GeneralCellGrowthData, OptimumParameters, calculate_aic, calculate_cell_growth_fitting, create_cell_growth_plot, calculate_cell_growth_optimum_parameters, growth_func
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import curve_fit

p0 = (0.1, 0.12)
extension = 20

app_ui = ui.page_fluid(
    ui.panel_title(ui.HTML('<div style="text-align: center; font-size: 30px;">'
                           '<span style="color: #7570B3;">G</span>'
                           '<span style="color: #D95F02;">r</span>'
                           '<span style="color: #EFCA08;">U</span>'
                           '<span style="color: #7570B3;">p</span>'
                           '<span style="color: #7A9A4E;">S</span>'
                           '<span style="color: #D95F02;">e</span>'
                           '</div>')),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file("upload_cell_growth_csv", "Upload CSV File (cell-growth data)", accept=[".csv"], multiple=False),
            ui.input_numeric("lwr_bound", "Lower bound for fit", 0),
            ui.input_numeric("upr_bound", "Upper bound for fit", 60),
            ui.input_checkbox("use_extrapolation", "Extrapolate", value=True),
            ui.output_text_verbatim("print_fitting_params"),
            ui.input_checkbox("use_ransac_regression_cell_growth", "Use RANSAC-regression for cell growth", False),
            ui.output_ui("cell_growth_data_uploaded"),
            ui.output_ui("metabolite_data_uploaded"),
        ),
        ui.panel_main(
            ui.output_plot("plot_cell_graph_normal_scale"),
            ui.output_ui("plot_cell_graph_normal_scale_download_conditional"),
            ui.output_plot("plot_cell_graph_log_scale"),
            ui.output_ui("plot_cell_graph_log_scale_download_conditional"),
            ui.output_plot("plot_metabolite_graph"),
            ui.output_ui("plot_metabolite_graph_download_conditional"),
        )  
    )
)

def server(input, output, session):
    general_cell_growth_data = reactive.Value(None)

    cell_graph_normal_scale_plot = reactive.Value(None)
    cell_graph_log_scale_plot = reactive.Value(None)
    metabolite_plot = reactive.Value(None)

    cell_growth_optimum_parameters = reactive.Value(None)

    @reactive.Calc
    def raw_cell_growth_df():
        csv_file = req(input.upload_cell_growth_csv())
        return pd.read_csv(csv_file[0]["datapath"])
    
    @reactive.Calc
    def use_extrapolation():
        return input.use_extrapolation()
    
    @reactive.Calc
    def use_ransac_regression_cell_growth():
        return input.use_ransac_regression_cell_growth()

    @reactive.Effect
    def cell_growth_general_properties():
        csv_df = raw_cell_growth_df()
        lower_bound, upper_bound = input.lwr_bound(), input.upr_bound()
        p0 = (0.1, 0.12)

        general_cell_growth_data.set(GeneralCellGrowthData(
            df=csv_df,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            p0=p0
        ))
    
    @reactive.Effect
    def cell_growth_fitting():
        cell_growth_data = req(general_cell_growth_data())

        cell_growth_fitting = calculate_cell_growth_fitting(
            cell_growth_data=cell_growth_data,
            use_ransac_regression=input.use_ransac_regression_cell_growth(),
            use_extrapolation=input.use_extrapolation()
        )

        cell_graph_normal_scale_plot.set(create_cell_growth_plot(
            fitting=cell_growth_fitting
        ))

        cell_graph_log_scale_plot.set(create_cell_growth_plot(
            fitting=cell_growth_fitting,
            use_log_scale=True
        ))

        cell_growth_optimum_parameters.set(calculate_cell_growth_optimum_parameters(cell_growth_fitting))

    @output
    @render.text
    def print_fitting_params():
        params = req(cell_growth_optimum_parameters())
        return "Cell growth parameters ± error\n" \
               f"μ ={params.b:.9f}±{params.confidence_interval_b:.9f} h⁻¹ \n" \
               f"X₀={params.a0:.9f}±{params.confidence_interval_a0:.9f} 10⁶cells·mL⁻¹\n" \
               f"R²={params.r_squared:.9f}"

    @output
    @render.plot
    def plot_cell_graph_normal_scale():
        return cell_graph_normal_scale_plot()
    
    @output
    @render.ui
    def plot_cell_graph_normal_scale_download_conditional():
        if raw_cell_growth_df() is not None:
            return ui.download_button("plot_cell_graph_normal_scale_download", "Download Normal Scale Graph")
        return None
    
    @output
    @render.download(filename="cell_graph_normal_scale.png")
    def plot_cell_graph_normal_scale_download():
        fig = cell_graph_normal_scale_plot()
        if fig is None:
            return None
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    @output
    @render.plot
    def plot_cell_graph_log_scale():
        return cell_graph_log_scale_plot()
    
    @output
    @render.ui
    def plot_cell_graph_log_scale_download_conditional():
        if raw_cell_growth_df() is not None:
            return ui.download_button("plot_cell_graph_log_scale_download", "Download Log Scale Graph")
        return None
    
    @output
    @render.download(filename="cell_graph_log_scale.png")
    def plot_cell_graph_log_scale_download():
        fig = cell_graph_log_scale_plot()
        if fig is None:
            return None
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf             

    @output
    @render.ui
    def cell_growth_data_uploaded():
        if raw_cell_growth_df() is None:
            return None
        return ui.input_file("upload_metabolite_csv", "Upload CSV file (metabolite data)", accept=[".csv"])
    
    @reactive.Calc
    def metabolite_df():
        csv_file = req(input.upload_metabolite_csv())
        return pd.read_csv(csv_file[0]["datapath"])
    
    @output
    @render.ui
    def metabolite_data_uploaded():
        if metabolite_df() is None:
            return None
        return (
            ui.output_ui("metabolites_filter"),
            ui.input_checkbox("use_ransac_regression_metabolites", "Use RANSAC-regression for metabolites", False),
            ui.input_checkbox("fit_glutamine", "Fit decaying metabolite", False),
            ui.output_ui("glutamine_ui"),
            ui.input_checkbox("fit_ammonia", "Fit decay-product metabolite", False),
            ui.output_ui("ammonia_ui"),
            ui.input_checkbox("convert_rate", "Convert metabolite rates to [mmol·gDW⁻¹h⁻¹]", False),
            ui.input_numeric("dry_mass_per_cell", "Dry mass per cell [pg]", value=250, min=0.001),
            ui.output_text_verbatim("dynamic_params_output"),
            ui.download_button("download_params", "Download Fitting Data")
        )
    
    @reactive.Calc
    def use_ransac_regression_metabolites():
        return input.use_ransac_regression_metabolites()
    
    @reactive.Calc
    def fit_glutamine():
        return input.fit_glutamine()
    
    @reactive.Calc
    def fit_ammonia():
        return input.fit_ammonia()
    
    @reactive.Calc
    def convert_rate():
        return input.convert_rate()
    
    @reactive.Calc
    def dry_mass_per_cell():
        return input.dry_mass_per_cell()
    
    @reactive.Calc
    def conversion_factor():
        if convert_rate():
            return (1e3) / (input.dry_mass_per_cell() )
        else:
            return 1
    
    @output
    @render.ui
    def metabolites_filter():
        if metabolite_df() is None: 
            return None
        columns = list(metabolite_df().columns)
        checkbox_filter = [ui.input_checkbox(f"chk_{col}", col, True) for col in columns[1:]]
        return ui.TagList(*checkbox_filter)
    
    @reactive.Calc
    def selected_metabolite_filter():
        if metabolite_df() is None:
            return []
        columns = list(metabolite_df().columns)
        selected = [col for col in columns[1:] if getattr(input, f"chk_{col}")()]
        return selected
        
    @output
    @render.ui
    def glutamine_ui():
        if not fit_glutamine() or metabolite_df() is None:
            return ui.TagList()
        columns = list(metabolite_df().columns)
        return ui.TagList(
            ui.input_select("glutamine_column", "Select Column", columns[1:]),
            ui.input_numeric("glutamine_k_value", "Enter decay constant k in [h⁻¹]", value=0.0025)
        )
    
    @reactive.Calc
    def glutamine_column():
        return input.glutamine_column()
    
    @reactive.Calc
    def glutamine_k_value():
        return input.glutamine_k_value()
    
    @output
    @render.ui
    def ammonia_ui():
        if not fit_ammonia() or metabolite_df() is None:
            return ui.TagList()
        columns = list(filter(lambda column: column != glutamine_column(), list(metabolite_df().columns)))
        return ui.TagList(
            ui.input_select("ammonia_column", "Select Column", columns[1:]),
        )

    @reactive.Calc
    def ammonia_column():
        return input.ammonia_column()

    dynamic_output_text = reactive.Value("")

    @reactive.Effect
    def create_metabolite_graph():
        global I, N

        if metabolite_df() is None:
            return None
        
        df = metabolite_df()

        fig, ax = plt.subplots(figsize=(10, 6))        
        columns = list(df.columns)
        selected_columns = selected_metabolite_filter()
        data_met = df.fillna(-1).values.astype(float)
        x_time = data_met[:, 0]
        lower_bound = general_cell_growth_data().lower_bound
        upper_bound = general_cell_growth_data().upper_bound
        use_extension = use_extrapolation()

        dynamic_outputs = []

        fitting_colors = ['#7570B3', '#90D7FF', '#66002C', '#F49FBC', '#EFCA08', '#E39EC1']
        color_cycle = itertools.cycle(fitting_colors)

        for column in selected_columns:
            i = columns.index(column)

            df_met = pd.DataFrame({'xTime': x_time, f'yMet{i}': data_met[:, i]})
            df_met_full = df_met[df_met[f'yMet{i}'] != -1]
            df_met_fit = df_met_full[(df_met_full['xTime'] >= lower_bound) & (df_met_full['xTime'] <= upper_bound)]

            fit_color = next(color_cycle)

            ax.scatter(df_met_full['xTime'], df_met_full[f'yMet{i}'], label=column, alpha=0.5, color='#1B9E77')
            ax.scatter(df_met_fit['xTime'], df_met_fit[f'yMet{i}'], color=fit_color, label=f'{column} (fit data)', alpha=0.5)
            
            x_fit = df_met_fit['xTime']
            y_fit = df_met_fit[f'yMet{i}']

            try:
                params: OptimumParameters = cell_growth_optimum_parameters()

                if use_ransac_regression_metabolites():
                    ransac = RANSACRegressor()
                    ransac.fit(x_fit.values.reshape(-1, 1), y_fit)
                    inlier_mask = ransac.inlier_mask_
                    outlier_mask = np.logical_not(inlier_mask)
                    x_fit = x_fit[inlier_mask]
                    y_fit = y_fit[inlier_mask]
                    ax.scatter(df_met_fit['xTime'][outlier_mask], df_met_fit[f'yMet{i}'][outlier_mask], color='#A2AAAD', label=f'{column} (outliers)', alpha=0.5)

                
                if fit_glutamine() and column == glutamine_column():
                    k_value = glutamine_k_value()
                elif fit_ammonia() and column == ammonia_column():
                    k_value = glutamine_k_value()  # Using the same k value for ammonia as for glutamine
                else:
                    k_value = 0  # Default k value for other metabolites

                A, B, q_g, P_0g = 1, 0, 0, 0
                if fit_ammonia() and column == ammonia_column():
                    q_g = I
                    P_0g = N
                    A, B = 0, 1                    

                popt, pcov = curve_fit(lambda t, q, P_0: growth_func(t, q, P_0, params.a0, params.b, k=k_value, A=A, B=B, q_g=q_g, P_0g=P_0g), x_fit, y_fit, p0=p0, bounds=(-200, 200))
                y_pred = growth_func(x_fit, *popt, params.a0, params.b, k=k_value, A=A, B=B, q_g=q_g, P_0g=P_0g)
                r_squared = r2_score(y_fit, y_pred)
                perr = np.sqrt(np.diag(pcov))
                confidence_intervals = 1.96 * perr
                x_fit_all = np.linspace(min(df_met_full['xTime']), max(df_met_full['xTime']) + (extension if use_extension else 0), 100)
                x_fit = x_fit_all[(x_fit_all >= lower_bound) & (x_fit_all <= upper_bound + (extension if use_extension else 0))]
                ax.plot(x_fit, growth_func(x_fit, *popt, params.a0, params.b, k=k_value, A=A, B=B, q_g=q_g, P_0g=P_0g), label=f'fit {column}', color='#D95F02')
                
                # Apply conversion factor only for display
                q = popt[0] * conversion_factor()
                q_ci = confidence_intervals[0] * conversion_factor()
                P_0 = popt[1]
                P_0_ci = confidence_intervals[1]
                
                # Calculate AIC
                residuals = y_fit - y_pred
                mse = np.mean(residuals**2)
                num_params = len(popt)
                aic = calculate_aic(len(y_fit), mse, num_params)
                
                # Save the optimization results for glutamine
                if fit_glutamine() and column == glutamine_column():
                    I, N = popt[0], P_0
                
                q_unit = "mmol·gDW⁻¹h⁻¹" if convert_rate() else "mmol·10⁹cells⁻¹·h⁻¹"
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
        
        metabolite_plot.set(fig)

    @output
    @render.text
    def dynamic_params_output():
        return dynamic_output_text()

    @output
    @render.plot
    def plot_metabolite_graph():
        return metabolite_plot()
                
    @output
    @render.ui
    def plot_metabolite_graph_download_conditional():
        if metabolite_df() is not None:
            return ui.download_button("plot_metabolite_graph_download", "Download Metabolite Graph")
        return None
    
    @output
    @render.download(filename="metabolite_graph.png")
    def plot_metabolite_graph_download():
        if metabolite_df() is None:
            return None
        
        fig = metabolite_plot()
        if fig is None:
            return None
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    @render.download(filename="fitting_parameters.txt")
    def download_params():
        if metabolite_df() is None:
            return None
        
        yield f"Uploaded Cell Growth File: {input.upload_cell_growth_csv()[0]['name']}\n"

        metabolite_input = input.upload_metabolite_csv()
        metabolite_files = []
        if metabolite_input:
            for file in metabolite_input:
                metabolite_files.append(file['name'])
        file_string = ', '.join(metabolite_files)

        yield f"Uploaded Metabolite File(s): {file_string}\n\n"
        yield f"Lower Bound: {general_cell_growth_data().lower_bound}\n"
        yield f"Upper Bound: {general_cell_growth_data().upper_bound}\n"
        yield f"RANSAC-regression used for cell growth: {use_ransac_regression_cell_growth()}\n\n"

        params = cell_growth_optimum_parameters()
        yield "Cell growth parameters ± error\n" \
            f"μ ={params.b:.9f}±{params.confidence_interval_b:.9f} h⁻¹ \n" \
            f"X₀={params.a0:.9f}±{params.confidence_interval_a0:.9f} 10⁶cells·mL⁻¹\n" \
            f"R²={params.r_squared:.9f}\n\n"
        
        selected_metabolites_list = selected_metabolite_filter()
        selected_metabolites_string = ', '.join(selected_metabolites_list)
        yield f"Selected Metabolites: {selected_metabolites_string}\n"
        yield f"RANSAC-regression used for metabolites: {use_ransac_regression_metabolites()}\n\n"

        if fit_glutamine():
            yield "Decaying Metabolite Fitted: True\n"
            yield f"Selected Metabolite: {glutamine_column()}\n"
            yield f"Selected Decay Constant K in [h⁻¹]: {glutamine_k_value()}\n\n"

        if fit_ammonia():
            yield "Decay-Product Metabolite Fitted: True\n"
            yield f"Selected Decay-Product Metabolite: {ammonia_column()}\n\n"

        yield f"Metabolite Rate Conversion to [mmol·gDW⁻¹h⁻¹] Selected: {convert_rate()}\n"
        yield f"Dry Mass Per Cell [pg]: {dry_mass_per_cell()}\n\n"

        yield dynamic_output_text()

app = App(app_ui, server)
