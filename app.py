#!/usr/bin/env python3
"""
Flask application for Abuja Traffic Analysis System.
Handles web interface, data entry, analysis, and report generation using ReportLab and Matplotlib.
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
from flask import Flask, render_template, send_file, request, redirect, url_for, flash, jsonify
import pandas as pd
import os
import logging
import numpy as np
from datetime import datetime
import tempfile
import shutil
from pathlib import Path
from traffic_analysis_models import TrafficAnalysisModel, EmissionModelType

app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'abuja-traffic-analysis-secret-key')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants aligned with traffic_analysis_models.py (updated for 2026)
VALUE_PER_MINUTE = 8.8
FUEL_PRICE_PETROL = 1045.0
FUEL_PRICE_DIESEL = 1210.0
EMISSION_FACTOR_PETROL = 2.31
EMISSION_FACTOR_DIESEL = 2.68
CORRIDOR_LENGTH = 6.0

# Define vehicle classes
vehicle_classes = [f'Class {i}' for i in range(1, 11)]  # Class 1 to Class 10

# Utility function to convert numpy types to native Python types for JSON serialization
def convert_numpy_types(obj):
    """Convert numpy data types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif pd.isna(obj):
        return None
    return obj

# Custom filters for formatting
@app.template_filter('format_number')
def format_number(value):
    """Format numbers with commas for readability."""
    try:
        if pd.isna(value) or value is None:
            return "0"
        return "{:,}".format(int(float(value)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for number formatting: {value}")
        return "0"

@app.template_filter('format_float')
def format_float(value):
    """Format float values to one decimal place."""
    try:
        if pd.isna(value) or value is None:
            return "0.0"
        return "{:,.1f}".format(float(value))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for float formatting: {value}")
        return "0.0"

@app.template_filter('format_currency')
def format_currency(value):
    """Format currency values with Naira symbol."""
    try:
        if pd.isna(value) or value is None:
            return "₦0.00"
        return "₦{:,.2f}".format(float(value))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for currency formatting: {value}")
        return "₦0.00"

def get_analysis_results(emission_model=EmissionModelType.BASIC):
    """Perform traffic analysis and generate results."""
    csv_file_path = "traffic_data.csv"
    try:
        if Path(csv_file_path).exists():
            logger.info(f"Loading traffic data from {csv_file_path}")
            model = TrafficAnalysisModel(csv_file_path=csv_file_path, emission_model=emission_model)
        else:
            logger.warning(f"CSV file {csv_file_path} not found. Using hardcoded data.")
            model = TrafficAnalysisModel(emission_model=emission_model)
    except Exception as e:
        logger.error(f"Error loading data: {e}. Using hardcoded data.", exc_info=True)
        model = TrafficAnalysisModel(emission_model=emission_model)

    logger.info(f"Running traffic analysis with {emission_model.value} model...")
    results = model.analyze_all_roads()
    if not results or not results['road_results']:
        logger.error("Analysis returned no results")
        raise ValueError("Traffic analysis failed to produce results")

    report_df = model.generate_report()
    report_df = report_df.fillna(0)

    # Convert numpy types to native Python types for JSON serialization
    for col in report_df.columns:
        if report_df[col].dtype == 'int64':
            report_df[col] = report_df[col].astype(int)
        elif report_df[col].dtype == 'float64':
            report_df[col] = report_df[col].astype(float)

    report_df.to_csv('abuja_traffic_analysis_report.csv', index=False)
    logger.info("Generated report saved to abuja_traffic_analysis_report.csv")

    vehicle_distributions = {road: model.get_vehicle_distribution(road) for road in model.road_data}
    chart_images = model.generate_chart_images()  # Generates temporary PNG files
    results = convert_numpy_types(results)

    # Convert report data to proper numeric types for template rendering
    report_data = report_df.to_dict('records')
    for row in report_data:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                continue
            try:
                if '.' in str(value):
                    row[key] = float(value)
                else:
                    row[key] = int(value)
            except (ValueError, TypeError):
                pass  # Keep as string if conversion fails

    return results, report_data, vehicle_distributions, chart_images, emission_model.value, model

# Routes
@app.route('/')
def welcome():
    """Render the welcome page."""
    return render_template('welcome.html', current_year=datetime.now().year)

@app.route('/data_entry', methods=['GET', 'POST'])
def data_entry():
    """Handle data entry form and save traffic data with new vehicle classes."""
    roads = ["Nyanya Road", "Lugbe Road", "Kubwa Road"]
    model = TrafficAnalysisModel()
    emission_models = model.get_available_models()
    current_data = {}
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')

    try:
        if Path('traffic_data.csv').exists():
            df = pd.read_csv('traffic_data.csv')
            logger.debug(f"Loaded traffic_data.csv with {len(df)} rows")
            for _, row in df.iterrows():
                current_data[(row['Road'], row['Vehicle Type'])] = row['Real_Vehicle_Count']
                current_data.update({
                    'entry_date': row.get('Date', current_date),
                    'entry_time': row.get('Time', current_time),
                    'congested_travel_time': row.get('Congested_Travel_Time_Minutes', 45.0),
                    'distance_km': row.get('Distance_KM', CORRIDOR_LENGTH),
                    'free_flow_time': row.get('Free_Flow_Time_Minutes', 4.0),
                    'value_per_minute': row.get('Value_Per_Minute_Naira', VALUE_PER_MINUTE),
                    'fuel_cost_per_liter_diesel': row.get('Fuel_Cost_Per_Liter_Diesel', FUEL_PRICE_DIESEL),
                    'fuel_cost_per_liter_petrol': row.get('Fuel_Cost_Per_Liter_Petrol', FUEL_PRICE_PETROL),
                    'emission_factor_petrol': row.get('Emission_Factor_Petrol', EMISSION_FACTOR_PETROL),
                    'emission_factor_diesel': row.get('Emission_Factor_Diesel', EMISSION_FACTOR_DIESEL),
                    'free_flow_speed': row.get('Free_Flow_Speed_KPH', 60.0),
                    'congested_speed': row.get('Congested_Speed_KPH', 8.0),
                    'avg_acceleration': row.get('Avg_Acceleration', 0.5),
                    'avg_deceleration': row.get('Avg_Deceleration', 0.5),
                    'idle_time_percentage': row.get('Idle_Time_Percentage', 0.3),
                    'emission_model': row.get('Emission_Model', EmissionModelType.BASIC.value),
                    'barth_alpha': row.get('Barth_Alpha', 0.0003),
                    'barth_beta': row.get('Barth_Beta', 0.0025),
                    'barth_gamma': row.get('Barth_Gamma', 0.0018),
                    'barth_traffic_flow': row.get('Barth_Traffic_Flow', 0.15),
                    'barth_road_gradient': row.get('Barth_Road_Gradient', 0.10),
                    'barth_acceleration': row.get('Barth_Acceleration', 0.04)
                })
                for vehicle_class in vehicle_classes:
                    field_name = f"emission_factor_{vehicle_class.replace(' ', '_')}"
                    current_data[field_name] = row.get('Vehicle_Emission_Factor',
                                                      model.VEHICLE_EMISSION_FACTORS.get(vehicle_class, 0.2))
        else:
            for road, vehicles in model.road_data.items():
                for vehicle_class in vehicle_classes:
                    current_data[(road, vehicle_class)] = vehicles.get(vehicle_class, 0)
            current_data.update({
                'entry_date': current_date,
                'entry_time': current_time,
                'congested_travel_time': 45.0,
                'distance_km': CORRIDOR_LENGTH,
                'free_flow_time': 4.0,
                'value_per_minute': VALUE_PER_MINUTE,
                'fuel_cost_per_liter_diesel': FUEL_PRICE_DIESEL,
                'fuel_cost_per_liter_petrol': FUEL_PRICE_PETROL,
                'emission_factor_petrol': EMISSION_FACTOR_PETROL,
                'emission_factor_diesel': EMISSION_FACTOR_DIESEL,
                'free_flow_speed': 60.0,
                'congested_speed': 8.0,
                'avg_acceleration': 0.5,
                'avg_deceleration': 0.5,
                'idle_time_percentage': 0.3,
                'emission_model': EmissionModelType.BASIC.value,
                'barth_alpha': 0.0003,
                'barth_beta': 0.0025,
                'barth_gamma': 0.0018,
                'barth_traffic_flow': 0.15,
                'barth_road_gradient': 0.10,
                'barth_acceleration': 0.04
            })
            for vehicle_class in vehicle_classes:
                current_data[f'emission_factor_{vehicle_class.replace(" ", "_")}'] = \
                    model.VEHICLE_EMISSION_FACTORS.get(vehicle_class, 0.2)
    except Exception as e:
        logger.error(f"Error loading current data: {str(e)}", exc_info=True)
        for road in roads:
            for vehicle_class in vehicle_classes:
                current_data[(road, vehicle_class)] = 0
        current_data.update({
            'entry_date': current_date,
            'entry_time': current_time,
            'congested_travel_time': 45.0,
            'distance_km': CORRIDOR_LENGTH,
            'free_flow_time': 4.0,
            'value_per_minute': VALUE_PER_MINUTE,
            'fuel_cost_per_liter_diesel': FUEL_PRICE_DIESEL,
            'fuel_cost_per_liter_petrol': FUEL_PRICE_PETROL,
            'emission_factor_petrol': EMISSION_FACTOR_PETROL,
            'emission_factor_diesel': EMISSION_FACTOR_DIESEL,
            'free_flow_speed': 60.0,
            'congested_speed': 8.0,
            'avg_acceleration': 0.5,
            'avg_deceleration': 0.5,
            'idle_time_percentage': 0.3,
            'emission_model': EmissionModelType.BASIC.value,
            'barth_alpha': 0.0003,
            'barth_beta': 0.0025,
            'barth_gamma': 0.0018,
            'barth_traffic_flow': 0.15,
            'barth_road_gradient': 0.10,
            'barth_acceleration': 0.04
        })
        for vehicle_class in vehicle_classes:
            current_data[f'emission_factor_{vehicle_class.replace(" ", "_")}'] = \
                model.VEHICLE_EMISSION_FACTORS.get(vehicle_class, 0.2)

    if request.method == 'POST':
        try:
            entry_date = request.form.get('entry_date', current_date)
            entry_time = request.form.get('entry_time', current_time)
            congested_travel_time = float(request.form.get('congested_travel_time', 45.0))
            distance_km = float(request.form.get('distance_km', CORRIDOR_LENGTH))
            free_flow_time = float(request.form.get('free_flow_time', 4.0))
            value_per_minute = float(request.form.get('value_per_minute', VALUE_PER_MINUTE))
            fuel_cost_per_liter_diesel = float(request.form.get('fuel_cost_per_liter_diesel', FUEL_PRICE_DIESEL))
            fuel_cost_per_liter_petrol = float(request.form.get('fuel_cost_per_liter_petrol', FUEL_PRICE_PETROL))
            emission_factor_petrol = float(request.form.get('emission_factor_petrol', EMISSION_FACTOR_PETROL))
            emission_factor_diesel = float(request.form.get('emission_factor_diesel', EMISSION_FACTOR_DIESEL))
            free_flow_speed = float(request.form.get('free_flow_speed', 60.0))
            congested_speed = float(request.form.get('congested_speed', 8.0))
            avg_acceleration = float(request.form.get('avg_acceleration', 0.5))
            avg_deceleration = float(request.form.get('avg_deceleration', 0.5))
            idle_time_percentage = float(request.form.get('idle_time_percentage', 0.3))
            emission_model = request.form.get('emission_model', EmissionModelType.BASIC.value)
            barth_alpha = float(request.form.get('barth_alpha', 0.0003))
            barth_beta = float(request.form.get('barth_beta', 0.0025))
            barth_gamma = float(request.form.get('barth_gamma', 0.0018))
            barth_traffic_flow = float(request.form.get('barth_traffic_flow', 0.15))
            barth_road_gradient = float(request.form.get('barth_road_gradient', 0.10))
            barth_acceleration = float(request.form.get('barth_acceleration', 0.04))
            vehicle_emission_factors = {}
            for vehicle_class in vehicle_classes:
                field_name = f"emission_factor_{vehicle_class.replace(' ', '_')}"
                factor = float(request.form.get(field_name, model.VEHICLE_EMISSION_FACTORS.get(vehicle_class, 0.2)))
                vehicle_emission_factors[vehicle_class] = factor

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
                for vehicle_class in vehicle_classes:
                    field_name = f"{road.replace(' ', '_')}_class{vehicle_class.split()[1]}_volume"
                    count = int(request.form.get(field_name, 0))
                    occupancy_rate = model.vehicle_parameters.get(f'class{vehicle_class.split()[1]}', {}).get('occupancy_avg', 1.0)
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
                        "Vehicle Type": vehicle_class,
                        "Real_Vehicle_Count": count,
                        "Real_VOR": occupancy_rate,
                        "Real_Delay_Time": congested_travel_time - free_flow_time,
                        "Free_Flow_Speed_KPH": free_flow_speed,
                        "Congested_Speed_KPH": congested_speed,
                        "Avg_Acceleration": avg_acceleration,
                        "Avg_Deceleration": avg_deceleration,
                        "Idle_Time_Percentage": idle_time_percentage,
                        "Emission_Model": emission_model,
                        "Barth_Alpha": barth_alpha,
                        "Barth_Beta": barth_beta,
                        "Barth_Gamma": barth_gamma,
                        "Barth_Traffic_Flow": barth_traffic_flow,
                        "Barth_Road_Gradient": barth_road_gradient,
                        "Barth_Acceleration": barth_acceleration,
                        "Vehicle_Emission_Factor": vehicle_emission_factors.get(vehicle_class, 0.2)
                    })

            df = pd.DataFrame(data)
            df.to_csv('traffic_data.csv', index=False)
            df.to_csv('traffic_data_backup.csv', index=False)
            logger.info("Saved form data to traffic_data.csv and traffic_data_backup.csv")
            flash("Data submitted successfully!", "success")
            return redirect(url_for('analysis'))
        except Exception as e:
            logger.error(f"Error processing form data: {str(e)}", exc_info=True)
            flash(f"Error processing input: {str(e)}", "error")
            return redirect(url_for('data_entry'))

    return render_template('data_entry.html',
                           roads=roads,
                           vehicle_types=vehicle_classes,
                           current_data=current_data,
                           emission_models=emission_models,
                           current_year=datetime.now().year)

@app.route('/analysis')
def analysis():
    """Render the analysis page with traffic data and charts."""
    if not Path('templates/pdf_report.html').exists():
        logger.error("pdf_report.html template not found")
        flash("Analysis template not found", "error")
        return redirect(url_for('data_entry'))

    try:
        emission_model_str = request.args.get('emission_model', EmissionModelType.BASIC.value)
        try:
            emission_model = EmissionModelType(emission_model_str)
        except ValueError:
            logger.warning(f"Invalid emission model: {emission_model_str}. Defaulting to BASIC.")
            emission_model = EmissionModelType.BASIC
        results, report_data, vehicle_distributions, chart_images, model_used, model = get_analysis_results(
            emission_model)
        summary = results['total_summary']
        road_results = results['road_results']
        emission_models = model.get_available_models()

        # Find the road with highest vehicle count
        highest_road_name = ""
        max_vehicles = 0
        for road_name, road_data in road_results.items():
            if road_data['total_vehicles'] > max_vehicles:
                max_vehicles = road_data['total_vehicles']
                highest_road_name = road_name

        # Prepare homepage data for pdf_report.html
        homepage_data = {
            'title': "Abuja Traffic Analysis System",
            'subtitle': "Comprehensive Traffic Congestion Analysis Report",
            'stats': [
                {'label': 'Total Vehicles Analyzed', 'value': f"{summary.get('total_vehicles_all_roads', 0):,}"},
                {'label': 'Total People Affected', 'value': f"{summary.get('total_people_all_roads', 0):,}"},
                {'label': 'Total CO₂ Emissions', 'value': f"{summary.get('total_co2_all_roads', 0):,.0f} kg"},
                {'label': 'Total Economic Impact',
                 'value': f"₦{summary.get('total_fuel_cost_all_roads', 0) + summary.get('total_productivity_loss_all_roads', 0):,.0f}"}
            ],
            'features': [
                "Accurate vehicle counting and classification",
                "Real-time congestion analysis",
                "Emission modeling (Basic, Barth, MOVES)",
                "Economic impact assessment"
            ],
            'how_it_works': [
                "Collect traffic data via sensors or manual input",
                "Analyze vehicle counts and congestion metrics",
                "Calculate fuel consumption and CO₂ emissions",
                "Generate comprehensive reports and visualizations"
            ]
        }

        return render_template(
            'pdf_report.html',
            homepage=homepage_data,
            summary=summary,
            road_results=road_results,
            report_data=report_data,
            vehicle_distributions=vehicle_distributions,
            chart_images={k: f"/temp_charts/{Path(v).name}" for k, v in chart_images.items()},
            emission_models=emission_models,
            current_emission_model=model_used,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
            current_year=datetime.now().year,
            results=results,
            highest_road_name=highest_road_name
        )
    except Exception as e:
        logger.error(f"Error in analysis route: {str(e)}", exc_info=True)
        flash(f"Error loading analysis: {str(e)}", "error")
        return redirect(url_for('data_entry'))

@app.route('/temp_charts/<filename>')
def serve_temp_chart(filename):
    """Serve temporary chart images for web display."""
    temp_dir = Path('temp_charts')
    file_path = temp_dir / filename
    if not file_path.exists():
        logger.error(f"Chart file not found: {file_path}")
        return "File not found", 404
    return send_file(file_path, mimetype='image/png')

@app.route('/analysis_data')
def analysis_data():
    """Provide JSON data for traffic_dashboard_pdf.html JavaScript rendering."""
    try:
        emission_model_str = request.args.get('emission_model', EmissionModelType.BASIC.value)
        try:
            emission_model = EmissionModelType(emission_model_str)
        except ValueError:
            logger.warning(f"Invalid emission model: {emission_model_str}. Defaulting to BASIC.")
            emission_model = EmissionModelType.BASIC
        results, report_data, vehicle_distributions, _, model_used, model = get_analysis_results(emission_model)
        return jsonify({
            'summary': convert_numpy_types(results['total_summary']),
            'road_results': convert_numpy_types(results['road_results']),
            'vehicle_distributions': convert_numpy_types(vehicle_distributions),
            'emission_model': model_used
        })
    except Exception as e:
        logger.error(f"Error in analysis_data route: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page,

 relying on client-side JavaScript."""
    if not Path('templates/traffic_dashboard_pdf.html').exists():
        logger.error("traffic_dashboard_pdf.html template not found")
        flash("Dashboard template not found", "error")
        return redirect(url_for('data_entry'))

    try:
        emission_model_str = request.args.get('emission_model', EmissionModelType.BASIC.value)
        try:
            emission_model = EmissionModelType(emission_model_str)
        except ValueError:
            logger.warning(f"Invalid emission model: {emission_model_str}. Defaulting to BASIC.")
            emission_model = EmissionModelType.BASIC
        model = TrafficAnalysisModel(emission_model=emission_model)
        emission_models = model.get_available_models()
        return render_template(
            'traffic_dashboard_pdf.html',
            emission_models=emission_models,
            current_emission_model=emission_model.value,
            current_year=datetime.now().year
        )
    except Exception as e:
        logger.error(f"Error in dashboard route: {str(e)}", exc_info=True)
        flash(f"Error loading dashboard: {str(e)}", "error")
        return redirect(url_for('data_entry'))

@app.route('/download_report')
def download_report():
    """Download the CSV report."""
    try:
        if not Path('abuja_traffic_analysis_report.csv').exists():
            raise FileNotFoundError("CSV report file not found")
        return send_file('abuja_traffic_analysis_report.csv', as_attachment=True)
    except Exception as e:
        logger.error(f"Error in download_report route: {str(e)}", exc_info=True)
        flash(f"Error downloading CSV report: {str(e)}", "error")
        return redirect(url_for('analysis'))

@app.route('/download_pdf')
def download_pdf():
    """Generate and download the PDF report using TrafficAnalysisModel's generate_pdf_report."""
    try:
        emission_model_str = request.args.get('emission_model', EmissionModelType.BASIC.value)
        try:
            emission_model = EmissionModelType(emission_model_str)
        except ValueError:
            logger.warning(f"Invalid emission model: {emission_model_str}. Defaulting to BASIC.")
            emission_model = EmissionModelType.BASIC

        results, report_data, vehicle_distributions, chart_images, model_used, model = get_analysis_results(
            emission_model)

        if not results or not report_data or not vehicle_distributions or not chart_images:
            logger.error("Incomplete analysis data for PDF generation")
            raise ValueError("Analysis data is incomplete or missing")

        highest_road_name = ""
        max_vehicles = 0
        for road_name, road_data in results['road_results'].items():
            if road_data['total_vehicles'] > max_vehicles:
                max_vehicles = road_data['total_vehicles']
                highest_road_name = road_name

        total_summary = results['total_summary']
        homepage_data = {
            'title': "Abuja Traffic Analysis System",
            'subtitle': "Comprehensive Traffic Congestion Analysis Report",
            'stats': [
                {'label': 'Total Vehicles Analyzed', 'value': f"{total_summary.get('total_vehicles_all_roads', 0):,}"},
                {'label': 'Total People Affected', 'value': f"{total_summary.get('total_people_all_roads', 0):,}"},
                {'label': 'Total CO₂ Emissions', 'value': f"{total_summary.get('total_co2_all_roads', 0):,.0f} kg"},
                {'label': 'Total Economic Impact',
                 'value': f"₦{total_summary.get('total_fuel_cost_all_roads', 0) + total_summary.get('total_productivity_loss_all_roads', 0):,.0f}"}
            ],
            'features': [
                "Accurate vehicle counting and classification",
                "Real-time congestion analysis",
                "Emission modeling (Basic, Barth, MOVES)",
                "Economic impact assessment"
            ],
            'how_it_works': [
                "Collect traffic data via sensors or manual input",
                "Analyze vehicle counts and congestion metrics",
                "Calculate fuel consumption and CO₂ emissions",
                "Generate comprehensive reports and visualizations"
            ],
            'methodology': {
                'description': "This assessment uses vehicle-specific parameters to calculate congestion impacts:",
                'parameters': [
                    f"Fuel Prices: ₦{report_data[0].get('Fuel_Cost_Per_Liter_Petrol', 1045.0):,.2f}/L (petrol), ₦{report_data[0].get('Fuel_Cost_Per_Liter_Diesel', 1210.0):,.2f}/L (diesel)",
                    f"Emission Factors: {report_data[0].get('Emission_Factor_Petrol', 2.31):,.2f} kg CO₂/L (petrol), {report_data[0].get('Emission_Factor_Diesel', 2.68):,.2f} kg CO₂/L (diesel)",
                    f"Productivity Value: ₦{report_data[0].get('Value_Per_Minute_Naira', 8.8):,.2f}/minute",
                    f"Corridor Length: {report_data[0].get('Distance_KM', 6.0):,.1f} km",
                    f"Time Parameters: Free flow: {report_data[0].get('Free_Flow_Time_Minutes', 4.0):,.1f} min, Congested: {report_data[0].get('Congested_Travel_Time_Minutes', 45.0):,.1f} min",
                    f"Advanced Parameters: Free flow speed: {report_data[0].get('Free_Flow_Speed_KPH', 60.0):,.1f} km/h, Congested speed: {report_data[0].get('Congested_Speed_KPH', 8.0):,.1f} km/h, Acceleration: {report_data[0].get('Avg_Acceleration', 0.5):,.1f} m/s², Idle time: {report_data[0].get('Idle_Time_Percentage', 0.3) * 100:,.1f}%"
                ] if model_used != 'basic' else [
                    f"Fuel Prices: ₦{report_data[0].get('Fuel_Cost_Per_Liter_Petrol', 1045.0):,.2f}/L (petrol), ₦{report_data[0].get('Fuel_Cost_Per_Liter_Diesel', 1210.0):,.2f}/L (diesel)",
                    f"Emission Factors: {report_data[0].get('Emission_Factor_Petrol', 2.31):,.2f} kg CO₂/L (petrol), {report_data[0].get('Emission_Factor_Diesel', 2.68):,.2f} kg CO₂/L (diesel)",
                    f"Productivity Value: ₦{report_data[0].get('Value_Per_Minute_Naira', 8.8):,.2f}/minute",
                    f"Corridor Length: {report_data[0].get('Distance_KM', 6.0):,.1f} km",
                    f"Time Parameters: Free flow: {report_data[0].get('Free_Flow_Time_Minutes', 4.0):,.1f} min, Congested: {report_data[0].get('Congested_Travel_Time_Minutes', 45.0):,.1f} min"
                ]
            },
            'recommendations': {
                'key_findings': [
                    f"{highest_road_name} shows the highest congestion impact, accounting for approximately {(results['road_results'][highest_road_name]['total_vehicles'] / total_summary['total_vehicles_all_roads'] * 100):,.1f}% of total vehicles and {(results['road_results'][highest_road_name]['total_productivity_loss_naira'] / total_summary['total_productivity_loss_all_roads'] * 100):,.1f}% of productivity losses.",
                    f"Evening rush hour congestion affects nearly {total_summary['total_people_all_roads']:,} commuters across all corridors.",
                    f"Total economic impact exceeds ₦{total_summary['total_fuel_cost_all_roads'] + total_summary['total_productivity_loss_all_roads']:,.0f} during the observed period.",
                    f"Environmental impact includes {total_summary['total_co2_all_roads']:,.0f} kg of excess CO₂ emissions.",
                    f"Average delay per vehicle is {(total_summary['total_delay_hours_all_roads'] * 60 / total_summary['total_vehicles_all_roads']):,.1f} minutes across all corridors.",
                    f"Analysis performed using {model_used.upper()} emission model for enhanced accuracy."
                ],
                'strategic_recommendations': [
                    f"Immediate Interventions: Implement targeted traffic management systems on {highest_road_name}, which shows the highest congestion metrics.",
                    "Public Transportation: Enhance alternative transportation options to reduce private vehicle numbers during peak hours.",
                    "Infrastructure Investment: Prioritize road improvements based on specific vehicle type distributions observed.",
                    "Policy Measures: Consider congestion pricing or staggered work hours to distribute traffic more evenly.",
                    "Data Collection: Expand monitoring to understand daily and weekly patterns beyond current snapshot data.",
                    f"Environmental Mitigation: Implement measures to offset {total_summary['total_co2_all_roads']:,.0f} kg of CO₂ emissions generated.",
                    f"Model Selection: Continue using {model_used.upper()} model for future analyses to maintain consistency in emission calculations."
                ]
            }
        }

        temp_dir = tempfile.mkdtemp()
        pdf_output = os.path.join(temp_dir, 'abuja_traffic_analysis_report.pdf')

        logger.info(f"Generating PDF report with {model_used} emission model")
        pdf_path = model.generate_pdf_report(
            output_path=pdf_output,
            homepage_data=homepage_data
        )

        if not pdf_path or not Path(pdf_path).exists():
            logger.error(f"PDF file was not created at {pdf_path}")
            raise Exception("PDF file was not created")

        response = send_file(
            pdf_path,
            as_attachment=True,
            download_name='abuja_traffic_analysis_report.pdf',
            mimetype='application/pdf'
        )

        @response.call_on_close
        def cleanup_temp_dir():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")

        flash("PDF report generated successfully!", "success")
        return response

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        flash(f"Failed to generate PDF: {str(e)}", "error")
        return redirect(url_for('analysis'))

@app.route('/download_traffic_data')
def download_traffic_data():
    """Download the backup traffic data CSV."""
    try:
        if not Path('traffic_data_backup.csv').exists():
            raise FileNotFoundError("Backup CSV file not found")
        return send_file('traffic_data_backup.csv', as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading traffic_data_backup.csv: {str(e)}", exc_info=True)
        flash(f"Error downloading backup data: {str(e)}", "error")
        return redirect(url_for('analysis'))

if __name__ == "__main__":
    print("Starting Abuja Traffic Analysis System...")
    print("Visit http://localhost:5000 to view the welcome page")
    print("Visit http://localhost:5000/analysis for traffic analysis")
    print("Visit http://localhost:5000/dashboard for the dashboard with PDF export")
    print("Visit http://localhost:5000/data_entry to input data")
    app.run(debug=True, host='0.0.0.0', port=5000)