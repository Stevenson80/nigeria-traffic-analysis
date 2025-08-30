from flask import Flask, render_template, send_file, request, redirect, url_for, flash, Response
import pandas as pd
import os
import logging
import io
import base64
from datetime import datetime
import pdfkit
from PyPDF2 import PdfMerger
from traffic_analysis_models import TrafficAnalysisModel, EmissionModelType

# Set Matplotlib backend before importing pyplot
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'abuja-traffic-analysis-secret-key')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('app.log'),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# wkhtmltopdf configuration - hardcoded for Render deployment
WKHTMLTOPDF_PATH = '/usr/bin/wkhtmltopdf'
if not os.path.exists(WKHTMLTOPDF_PATH):
    logger.warning(f"wkhtmltopdf not found at {WKHTMLTOPDF_PATH}. Ensure it is installed.")
config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)


# Custom filters for formatting
@app.template_filter('format_number')
def format_number(value):
    try:
        if pd.isna(value):
            return "0"
        return "{:,}".format(int(float(value)))
    except (ValueError, TypeError):
        return "0"


@app.template_filter('format_float')
def format_float(value):
    try:
        if pd.isna(value):
            return "0.0"
        return "{:,.1f}".format(float(value))
    except (ValueError, TypeError):
        return "0.0"


@app.template_filter('format_currency')
def format_currency(value):
    try:
        if pd.isna(value):
            return "₦0.00"
        return "₦{:,.2f}".format(float(value))
    except (ValueError, TypeError):
        return "₦0.00"


def file_to_base64(file_path):
    """Convert a file to base64-encoded string for web display."""
    try:
        with open(file_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        return f"data:image/png;base64,{img_data}"
    except Exception as e:
        logger.error(f"Error converting {file_path} to base64: {str(e)}")
        return ""


def get_analysis_results(emission_model=EmissionModelType.BASIC):
    """Run analysis and generate charts using TrafficAnalysisModel."""
    csv_file_path = "traffic_data.csv"
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')

    try:
        model = TrafficAnalysisModel(csv_file_path=csv_file_path if os.path.exists(csv_file_path) else None,
                                     emission_model=emission_model)
        logger.info(f"Running traffic analysis with {emission_model.value} model...")
        results = model.analyze_all_roads()
        logger.debug(f"Analysis completed with {len(results['road_results'])} roads")

        # Generate report
        report_df = model.generate_report()
        report_df = report_df.fillna(0)

        # Convert appropriate columns to integers
        for col in ['Vehicle Count', 'People Affected']:
            if col in report_df.columns:
                report_df[col] = report_df[col].astype(int)

        # Save report
        report_df.to_csv('abuja_traffic_analysis_report.csv', index=False)
        logger.info("Generated report saved to abuja_traffic_analysis_report.csv")

        # Prepare vehicle distributions for charts
        vehicle_distributions = {}
        for road_name, vehicle_data in model.road_data.items():
            vehicle_distributions[road_name] = [
                {"vehicle_type": vt, "count": int(count) if not pd.isna(count) else 0}
                for vt, count in vehicle_data.items() if count > 0
            ]

        # Generate charts using TrafficAnalysisModel
        chart_files = model.generate_charts()
        base_dir = os.path.abspath(os.path.dirname(__file__))
        chart_files = {k: os.path.join(base_dir, v) for k, v in chart_files.items()}

        # Convert chart files to base64 for web display
        chart_images = {k: file_to_base64(v) for k, v in chart_files.items()}

        # Verify chart files exist
        for chart_name, chart_path in chart_files.items():
            if not os.path.exists(chart_path):
                logger.warning(f"Chart file {chart_path} does not exist")

        # Ensure summary has all required fields for analysis.html
        summary = results.get('total_summary', {})
        summary_defaults = {
            'date': current_date,
            'time': current_time,
            'total_vehicles_all_roads': 0,
            'total_people_all_roads': 0,
            'total_delay_hours_all_roads': 0.0,
            'total_excess_fuel_all_roads': 0.0,
            'total_fuel_cost_all_roads': 0.0,
            'total_co2_all_roads': 0.0,
            'total_productivity_loss_all_roads': 0.0,
            'avg_fuel_cost_per_vehicle': 0.0,
            'Fuel_Cost_Per_Liter_Petrol': 850.0,
            'Fuel_Cost_Per_Liter_Diesel': 1000.0,
            'Emission_Factor_Petrol': 2.31,
            'Emission_Factor_Diesel': 2.68,
            'Value_Per_Minute_Naira': 7.29,
            'Distance_KM': 6.0,
            'Free_Flow_Time_Minutes': 4.0,
            'Congested_Travel_Time_Minutes': 45.0,
            'Free_Flow_Speed_KPH': 60.0,
            'Congested_Speed_KPH': 8.0,
            'Avg_Acceleration': 0.5,
            'Avg_Deceleration': 0.5,
            'Idle_Time_Percentage': 0.3
        }
        for key, value in summary_defaults.items():
            summary.setdefault(key, value)

        return results, report_df, vehicle_distributions, chart_files, chart_images, emission_model.value
    except Exception as e:
        logger.error(f"Error in get_analysis_results: {str(e)}")
        flash(f"Error running analysis: {str(e)}", "error")
        return {}, pd.DataFrame(), {}, {}, {}, emission_model.value


# Routes
@app.route('/')
def welcome():
    return render_template('welcome.html', current_year=datetime.now().year)


@app.route('/data_entry', methods=['GET', 'POST'])
def data_entry():
    roads = ["Nyanya Road", "Lugbe Road", "Kubwa Road"]
    vehicle_types = ['Motorcycles', 'Cars', 'SUVs', 'Sedans', 'Wagons',
                     'Short Buses', 'Minibusses', 'Long Buses', 'Truck', 'Tanker and Trailer']

    model = TrafficAnalysisModel()
    emission_models = model.get_available_models()

    current_data = {}
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')

    try:
        if os.path.exists('traffic_data.csv'):
            df = pd.read_csv('traffic_data.csv')
            logger.debug(f"Loaded traffic_data.csv with {len(df)} rows")
            for _, row in df.iterrows():
                current_data[(row['Road'], row['Vehicle Type'])] = row['Real_Vehicle_Count']
                if 'entry_date' not in current_data:
                    current_data['entry_date'] = row.get('Date', current_date)
                    current_data['entry_time'] = row.get('Time', current_time)
                    current_data['congested_travel_time'] = row.get('Congested_Travel_Time_Minutes', 45.0)
                    current_data['distance_km'] = row.get('Distance_KM', 6.0)
                    current_data['free_flow_time'] = row.get('Free_Flow_Time_Minutes', 4.0)
                    current_data['value_per_minute'] = row.get('Value_Per_Minute_Naira', 7.29)
                    current_data['fuel_cost_per_liter_diesel'] = row.get('Fuel_Cost_Per_Liter_Diesel', 1000.0)
                    current_data['fuel_cost_per_liter_petrol'] = row.get('Fuel_Cost_Per_Liter_Petrol', 850.0)
                    current_data['emission_factor_petrol'] = row.get('Emission_Factor_Petrol', 2.31)
                    current_data['emission_factor_diesel'] = row.get('Emission_Factor_Diesel', 2.68)
                    current_data['free_flow_speed'] = row.get('Free_Flow_Speed_KPH', 60.0)
                    current_data['congested_speed'] = row.get('Congested_Speed_KPH', 8.0)
                    current_data['avg_acceleration'] = row.get('Avg_Acceleration', 0.5)
                    current_data['avg_deceleration'] = row.get('Avg_Deceleration', 0.5)
                    current_data['idle_time_percentage'] = row.get('Idle_Time_Percentage', 0.3)
                    current_data['emission_model'] = row.get('Emission_Model', EmissionModelType.BASIC.value)

                    # Barth parameters
                    current_data['barth_alpha'] = row.get('Barth_Alpha', 0.0003)
                    current_data['barth_beta'] = row.get('Barth_Beta', 0.0025)
                    current_data['barth_gamma'] = row.get('Barth_Gamma', 0.0018)
                    current_data['barth_traffic_flow'] = row.get('Barth_Traffic_Flow', 0.15)
                    current_data['barth_road_gradient'] = row.get('Barth_Road_Gradient', 0.10)
                    current_data['barth_acceleration'] = row.get('Barth_Acceleration', 0.04)

                    # Vehicle-specific emission factors
                    for vehicle_type in vehicle_types:
                        field_name = f"emission_factor_{vehicle_type.replace(' ', '_').replace('/', '_')}"
                        current_data[field_name] = row.get('Vehicle_Emission_Factor',
                                                           model.VEHICLE_EMISSION_FACTORS.get(vehicle_type, 0.2))
        else:
            model = TrafficAnalysisModel()
            for road, vehicles in model.road_data.items():
                for vehicle_type, count in vehicles.items():
                    current_data[(road, vehicle_type)] = count
            current_data['entry_date'] = current_date
            current_data['entry_time'] = current_time
            current_data['congested_travel_time'] = 45.0
            current_data['distance_km'] = 6.0
            current_data['free_flow_time'] = 4.0
            current_data['value_per_minute'] = 7.29
            current_data['fuel_cost_per_liter_diesel'] = 1000.0
            current_data['fuel_cost_per_liter_petrol'] = 850.0
            current_data['emission_factor_petrol'] = 2.31
            current_data['emission_factor_diesel'] = 2.68
            current_data['free_flow_speed'] = 60.0
            current_data['congested_speed'] = 8.0
            current_data['avg_acceleration'] = 0.5
            current_data['avg_deceleration'] = 0.5
            current_data['idle_time_percentage'] = 0.3
            current_data['emission_model'] = EmissionModelType.BASIC.value

            # Add Barth parameters to current_data initialization
            current_data['barth_alpha'] = 0.0003
            current_data['barth_beta'] = 0.0025
            current_data['barth_gamma'] = 0.0018
            current_data['barth_traffic_flow'] = 0.15
            current_data['barth_road_gradient'] = 0.10
            current_data['barth_acceleration'] = 0.04

            # Add emission factors
            for vehicle_type in vehicle_types:
                current_data[f'emission_factor_{vehicle_type.replace(" ", "_").replace("/", "_")}'] = \
                    model.VEHICLE_EMISSION_FACTORS.get(vehicle_type, 0.2)
    except Exception as e:
        logger.error(f"Error loading current data: {str(e)}")
        for road in roads:
            for vehicle_type in vehicle_types:
                current_data[(road, vehicle_type)] = 0
        current_data['entry_date'] = current_date
        current_data['entry_time'] = current_time
        current_data['congested_travel_time'] = 45.0
        current_data['distance_km'] = 6.0
        current_data['free_flow_time'] = 4.0
        current_data['value_per_minute'] = 7.29
        current_data['fuel_cost_per_liter_diesel'] = 1000.0
        current_data['fuel_cost_per_liter_petrol'] = 850.0
        current_data['emission_factor_petrol'] = 2.31
        current_data['emission_factor_diesel'] = 2.68
        current_data['free_flow_speed'] = 60.0
        current_data['congested_speed'] = 8.0
        current_data['avg_acceleration'] = 0.5
        current_data['avg_deceleration'] = 0.5
        current_data['idle_time_percentage'] = 0.3
        current_data['emission_model'] = EmissionModelType.BASIC.value

        # Add Barth parameters to current_data initialization
        current_data['barth_alpha'] = 0.0003
        current_data['barth_beta'] = 0.0025
        current_data['barth_gamma'] = 0.0018
        current_data['barth_traffic_flow'] = 0.15
        current_data['barth_road_gradient'] = 0.10
        current_data['barth_acceleration'] = 0.04

        # Add emission factors
        for vehicle_type in vehicle_types:
            current_data[f'emission_factor_{vehicle_type.replace(" ", "_").replace("/", "_")}'] = \
                model.VEHICLE_EMISSION_FACTORS.get(vehicle_type, 0.2)

    if request.method == 'POST':
        try:
            entry_date = request.form.get('entry_date', current_date)
            entry_time = request.form.get('entry_time', current_time)
            congested_travel_time = float(request.form.get('congested_travel_time', 45.0))
            distance_km = float(request.form.get('distance_km', 6.0))
            free_flow_time = float(request.form.get('free_flow_time', 4.0))
            value_per_minute = float(request.form.get('value_per_minute', 7.29))
            fuel_cost_per_liter_diesel = float(request.form.get('fuel_cost_per_liter_diesel', 1000.0))
            fuel_cost_per_liter_petrol = float(request.form.get('fuel_cost_per_liter_petrol', 850.0))
            emission_factor_petrol = float(request.form.get('emission_factor_petrol', 2.31))
            emission_factor_diesel = float(request.form.get('emission_factor_diesel', 2.68))
            free_flow_speed = float(request.form.get('free_flow_speed', 60.0))
            congested_speed = float(request.form.get('congested_speed', 8.0))
            avg_acceleration = float(request.form.get('avg_acceleration', 0.5))
            avg_deceleration = float(request.form.get('avg_deceleration', 0.5))
            idle_time_percentage = float(request.form.get('idle_time_percentage', 0.3))
            emission_model = request.form.get('emission_model', EmissionModelType.BASIC.value)

            # Get Barth parameters
            barth_alpha = float(request.form.get('barth_alpha', 0.0003))
            barth_beta = float(request.form.get('barth_beta', 0.0025))
            barth_gamma = float(request.form.get('barth_gamma', 0.0018))
            barth_traffic_flow = float(request.form.get('barth_traffic_flow', 0.15))
            barth_road_gradient = float(request.form.get('barth_road_gradient', 0.10))
            barth_acceleration = float(request.form.get('barth_acceleration', 0.04))

            # Get vehicle-specific emission factors
            vehicle_emission_factors = {}
            for vehicle_type in vehicle_types:
                field_name = f"emission_factor_{vehicle_type.replace(' ', '_').replace('/', '_')}"
                factor = float(request.form.get(field_name, 0.2))
                vehicle_emission_factors[vehicle_type] = factor

            if congested_travel_time <= free_flow_time:
                flash("Congested travel time must be greater than free flow time", "error")
                return redirect(url_for('data_entry'))

            if any(x <= 0 for x in [distance_km, free_flow_time, value_per_minute,
                                    fuel_cost_per_liter_diesel, fuel_cost_per_liter_petrol,
                                    emission_factor_petrol, emission_factor_diesel,
                                    free_flow_speed, congested_speed, avg_acceleration,
                                    avg_deceleration, idle_time_percentage,
                                    barth_alpha, barth_beta, barth_gamma,
                                    barth_traffic_flow, barth_road_gradient, barth_acceleration]):
                flash("All values must be positive numbers", "error")
                return redirect(url_for('data_entry'))

            data = []
            for road in roads:
                for vehicle_type in vehicle_types:
                    field_name = f"{road.replace(' ', '_')}_{vehicle_type.replace(' ', '_').replace('/', '_')}"
                    count = int(request.form.get(field_name, 0))
                    model = TrafficAnalysisModel()
                    occupancy_rate = model.vehicle_parameters[vehicle_type]['occupancy_avg']
                    data.append({
                        "Date": entry_date,
                        "Time": entry_time,
                        "Congested_Travel_Time_Minutes": congested_travel_time,
                        "Distance_KM": distance_km,
                        "Free_Flow_Time_Minutes": free_flow_time,
                        "Value_Per_Minute_Naira": value_per_minute,
                        "Fuel_Cost_Per_Liter_Diesel": fuel_cost_per_liter_diesel,
                        "Fuel_Cost_Per_Liter_Petrol": fuel_cost_per_liter_petrol,
                        "Emission_Factor_Petrol": emission_factor_petrol,
                        "Emission_Factor_Diesel": emission_factor_diesel,
                        "Road": road,
                        "Vehicle Type": vehicle_type,
                        "Real_Vehicle_Count": count,
                        "Real_VOR": occupancy_rate,
                        "Real_Delay_Time": congested_travel_time - free_flow_time,
                        "Free_Flow_Speed_KPH": free_flow_speed,
                        "Congested_Speed_KPH": congested_speed,
                        "Avg_Acceleration": avg_acceleration,
                        "Avg_Deceleration": avg_deceleration,
                        "Idle_Time_Percentage": idle_time_percentage,
                        "Emission_Model": emission_model,

                        # Barth parameters
                        "Barth_Alpha": barth_alpha,
                        "Barth_Beta": barth_beta,
                        "Barth_Gamma": barth_gamma,
                        "Barth_Traffic_Flow": barth_traffic_flow,
                        "Barth_Road_Gradient": barth_road_gradient,
                        "Barth_Acceleration": barth_acceleration,

                        # Vehicle-specific emission factor
                        "Vehicle_Emission_Factor": vehicle_emission_factors.get(vehicle_type, 0.2)
                    })

            df = pd.DataFrame(data)
            df.to_csv('traffic_data.csv', index=False)
            df.to_csv('traffic_data_backup.csv', index=False)
            logger.info("Saved form data to traffic_data.csv and traffic_data_backup.csv")
            flash("Data submitted successfully!", "success")
            return redirect(url_for('analysis'))
        except Exception as e:
            logger.error(f"Error processing form data: {str(e)}")
            flash(f"Error processing input: {str(e)}", "error")
            return redirect(url_for('data_entry'))

    return render_template('data_entry.html',
                           roads=roads,
                           vehicle_types=vehicle_types,
                           current_data=current_data,
                           emission_models=emission_models,
                           current_year=datetime.now().year)


@app.route('/analysis')
def analysis():
    try:
        emission_model_str = request.args.get('emission_model', EmissionModelType.BASIC.value)
        if not emission_model_str or emission_model_str not in [model.value for model in EmissionModelType]:
            emission_model_str = EmissionModelType.BASIC.value
            logger.warning(
                f"Invalid emission_model '{emission_model_str}', defaulting to {EmissionModelType.BASIC.value}")
        emission_model = EmissionModelType(emission_model_str)
        results, report_df, vehicle_distributions, chart_files, chart_images, model_used = get_analysis_results(
            emission_model)
        summary = results.get('total_summary', {})
        road_results = results.get('road_results', {})
        model = TrafficAnalysisModel()
        emission_models = model.get_available_models()
        return render_template(
            'analysis.html',
            summary=summary,
            road_results=road_results,
            report_data=report_df.to_dict('records'),
            vehicle_distributions=vehicle_distributions,
            chart_images=chart_images,
            emission_models=[model for model in EmissionModelType],
            current_emission_model=model_used,
            current_year=datetime.now().year
        )
    except Exception as e:
        logger.error(f"Error in analysis route: {str(e)}")
        flash(f"Error loading analysis: {str(e)}", "error")
        return redirect(url_for('data_entry'))


@app.route('/dashboard')
def dashboard():
    try:
        emission_model_str = request.args.get('emission_model', EmissionModelType.BASIC.value)
        if not emission_model_str or emission_model_str not in [model.value for model in EmissionModelType]:
            emission_model_str = EmissionModelType.BASIC.value
            logger.warning(
                f"Invalid emission_model '{emission_model_str}', defaulting to {EmissionModelType.BASIC.value}")
        emission_model = EmissionModelType(emission_model_str)
        results, report_df, vehicle_distributions, chart_files, chart_images, model_used = get_analysis_results(
            emission_model)
        summary = results.get('total_summary', {})
        road_results = results.get('road_results', {})
        model = TrafficAnalysisModel()
        emission_models = model.get_available_models()
        return render_template(
            'traffic_dashboard_pdf.html',
            summary=summary,
            road_results=road_results,
            report_data=report_df.to_dict('records'),
            vehicle_distributions=vehicle_distributions,
            chart_images=chart_images,
            emission_models=[model for model in EmissionModelType],
            current_year=datetime.now().year
        )
    except Exception as e:
        logger.error(f"Error in dashboard route: {str(e)}")
        flash(f"Error loading dashboard: {str(e)}", "error")
        return redirect(url_for('data_entry'))


@app.route('/download_report')
def download_report():
    try:
        if not os.path.exists('abuja_traffic_analysis_report.csv'):
            raise FileNotFoundError("CSV report file not found")
        return send_file('abuja_traffic_analysis_report.csv', as_attachment=True)
    except Exception as e:
        logger.error(f"Error in download_report route: {str(e)}")
        flash(f"Error downloading CSV report: {str(e)}", "error")
        return redirect(url_for('analysis'))


@app.route('/download_pdf')
def download_pdf():
    try:
        emission_model_str = request.args.get('emission_model', EmissionModelType.BASIC.value)
        if not emission_model_str or emission_model_str not in [model.value for model in EmissionModelType]:
            emission_model_str = EmissionModelType.BASIC.value
            logger.warning(
                f"Invalid emission_model '{emission_model_str}', defaulting to {EmissionModelType.BASIC.value}")
        emission_model = EmissionModelType(emission_model_str)
        results, report_df, vehicle_distributions, chart_files, chart_images, model_used = get_analysis_results(
            emission_model)
        summary = results.get('total_summary', {})
        road_results = results.get('road_results', {})

        # Render cover page
        cover_rendered = render_template(
            'pdf_cover.html',
            current_year=datetime.now().year
        )

        # Render main report
        report_rendered = render_template(
            'pdf_report.html',
            summary=summary,
            road_results=road_results,
            report_data=report_df.to_dict('records'),
            vehicle_distributions=vehicle_distributions,
            chart_files=chart_files,
            emission_model=model_used,
            current_year=datetime.now().year,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M')
        )

        # Save rendered HTML for debugging
        html_output_path = 'pdf_report_rendered.html'
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(cover_rendered + '\n' + report_rendered)
        logger.info(f"Saved rendered HTML to {html_output_path}")

        # Generate PDFs
        cover_pdf = 'cover.pdf'
        report_pdf = 'report.pdf'
        final_pdf = 'abuja_traffic_analysis_report.pdf'
        options = {
            'page-size': 'A4',
            'margin-top': '0.5in',
            'margin-right': '0.5in',
            'margin-bottom': '0.5in',
            'margin-left': '0.5in',
            'encoding': 'UTF-8',
            'enable-local-file-access': '',
            'quiet': ''
        }

        # Generate cover PDF
        pdfkit.from_string(cover_rendered, cover_pdf, configuration=config, options=options)
        logger.info(f"Generated cover PDF at {cover_pdf}")

        # Generate report PDF
        pdfkit.from_string(report_rendered, report_pdf, configuration=config, options=options)
        logger.info(f"Generated report PDF at {report_pdf}")

        # Merge PDFs
        merger = PdfMerger()
        merger.append(cover_pdf)
        merger.append(report_pdf)
        merger.write(final_pdf)
        merger.close()

        # Clean up temporary PDFs
        for temp_pdf in [cover_pdf, report_pdf]:
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)
                logger.info(f"Removed temporary PDF {temp_pdf}")

        if not os.path.exists(final_pdf) or os.path.getsize(final_pdf) == 0:
            logger.error("PDF file was not created or is empty")
            flash("PDF file was not created or is empty. Check wkhtmltopdf and file permissions.", "error")
            return Response("PDF generation failed: File not created or empty", status=500)

        flash("PDF report generated successfully!", "success")
        return send_file(final_pdf, as_attachment=True, download_name='abuja_traffic_analysis_report.pdf')
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        flash(f"Failed to generate PDF: {str(e)}", "error")
        return Response(f"PDF generation failed: {str(e)}", status=500)


@app.route('/download_traffic_data')
def download_traffic_data():
    try:
        if not os.path.exists('traffic_data_backup.csv'):
            raise FileNotFoundError("Backup CSV file not found")
        return send_file('traffic_data_backup.csv', as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading traffic_data_backup.csv: {str(e)}")
        flash(f"Error downloading backup data: {str(e)}", "error")
        return redirect(url_for('analysis'))


if __name__ == "__main__":
    print("Starting Abuja Traffic Analysis System...")
    print("Visit http://localhost:5000 to view the welcome page")
    print("Visit http://localhost:5000/analysis for traffic analysis")
    print("Visit http://localhost:5000/dashboard for the new dashboard with PDF export")
    print("Visit http://localhost:5000/data_entry to input data")
    app.run(debug=True, host='0.0.0.0', port=5000)