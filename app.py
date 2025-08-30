import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from flask import Flask, render_template, send_file, request, redirect, url_for, flash, Response
import pandas as pd
import os
import logging
import io
import base64
import numpy as np
from datetime import datetime
import pdfkit
from traffic_analysis_models import TrafficAnalysisModel, EmissionModelType

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'abuja-traffic-analysis-secret-key')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('app.log'),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# wkhtmltopdf configuration
WKHTMLTOPDF_PATH = os.environ.get('WKHTMLTOPDF_PATH', 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')
config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

def save_charts_for_pdf(road_results, vehicle_distributions):
    """Save chart images as temporary files for PDF generation."""
    chart_files = {}
    temp_dir = "temp_charts"

    try:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        # Clear previous temp files
        for file in os.listdir(temp_dir):
            if file.endswith('.png'):
                try:
                    os.remove(os.path.join(temp_dir, file))
                    logger.debug(f"Deleted temporary file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {file}: {str(e)}")

        # Save pie charts for each road
        for road_name, vehicles in vehicle_distributions.items():
            if vehicles and any(v['count'] > 0 for v in vehicles):
                labels = [v['vehicle_type'] for v in vehicles if v['count'] > 0]
                data = [v['count'] for v in vehicles if v['count'] > 0]

                plt.figure(figsize=(6, 6))
                colors = ['#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56', '#9966FF', '#FF4500', '#2E8B57']
                plt.pie(data, labels=labels, autopct='%1.1f%%', colors=colors[:len(data)], startangle=90)
                plt.title(f"{road_name} Vehicle Distribution", fontsize=12)
                plt.axis('equal')

                filename = os.path.join(temp_dir, f"pie_{road_name.replace(' ', '_')}.png")
                plt.savefig(filename, bbox_inches='tight', dpi=150)
                plt.close()
                chart_files[f"pie_{road_name.replace(' ', '_')}"] = filename
                logger.info(f"Saved pie chart: {filename}")

        # Save bar charts
        road_names = list(road_results.keys())
        if not road_names:
            logger.warning("No road names available for bar charts")
            return chart_files

        # People affected chart
        people_data = [road_results[r].get('total_people', 0) for r in road_names]
        if any(x > 0 for x in people_data):
            plt.figure(figsize=(8, 5))
            plt.bar(road_names, people_data, color=['#4BC0C0', '#FF6384', '#36A2EB'])
            plt.title('People Affected by Congestion', fontsize=14)
            plt.ylabel('People', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            for i, v in enumerate(people_data):
                if v > 0:
                    plt.text(i, v + max(people_data) * 0.01, f'{v:,.0f}', ha='center', fontsize=9)
            filename = os.path.join(temp_dir, "people_chart.png")
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()
            chart_files['people_chart'] = filename
            logger.info(f"Saved bar chart: {filename}")

        # Fuel consumption chart
        fuel_data = [road_results[r].get('total_excess_fuel_l', 0) for r in road_names]
        if any(x > 0 for x in fuel_data):
            plt.figure(figsize=(8, 5))
            plt.bar(road_names, fuel_data, color=['#4BC0C0', '#FF6384', '#36A2EB'])
            plt.title('Excess Fuel Consumption', fontsize=14)
            plt.ylabel('Liters', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            for i, v in enumerate(fuel_data):
                if v > 0:
                    plt.text(i, v + max(fuel_data) * 0.01, f'{v:,.1f}', ha='center', fontsize=9)
            filename = os.path.join(temp_dir, "fuel_chart.png")
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()
            chart_files['fuel_chart'] = filename
            logger.info(f"Saved bar chart: {filename}")

        # CO2 emissions chart
        co2_data = [road_results[r].get('total_co2_kg', 0) for r in road_names]
        if any(x > 0 for x in co2_data):
            plt.figure(figsize=(8, 5))
            plt.bar(road_names, co2_data, color=['#4BC0C0', '#FF6384', '#36A2EB'])
            plt.title('Excess CO₂ Emissions', fontsize=14)
            plt.ylabel('kg', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            for i, v in enumerate(co2_data):
                if v > 0:
                    plt.text(i, v + max(co2_data) * 0.01, f'{v:,.1f}', ha='center', fontsize=9)
            filename = os.path.join(temp_dir, "co2_chart.png")
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()
            chart_files['co2_chart'] = filename
            logger.info(f"Saved bar chart: {filename}")

        # Total cost chart
        cost_data = [(road_results[r].get('total_fuel_cost_naira', 0) +
                      road_results[r].get('total_productivity_loss_naira', 0)) for r in road_names]
        if any(x > 0 for x in cost_data):
            plt.figure(figsize=(8, 5))
            plt.bar(road_names, cost_data, color=['#4BC0C0', '#FF6384', '#36A2EB'])
            plt.title('Total Cost (Fuel + Productivity Loss)', fontsize=14)
            plt.ylabel('Cost (₦)', fontsize=12)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₦{x:,.0f}'))
            plt.xticks(rotation=45, ha='right')
            for i, v in enumerate(cost_data):
                if v > 0:
                    plt.text(i, v + max(cost_data) * 0.01, f'₦{v:,.0f}', ha='center', fontsize=9)
            filename = os.path.join(temp_dir, "cost_chart.png")
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()
            chart_files['cost_chart'] = filename
            logger.info(f"Saved bar chart: {filename}")

    except Exception as e:
        logger.error(f"Error saving charts for PDF: {str(e)}")
        flash(f"Error generating charts: {str(e)}", "error")

    return chart_files

# Custom filters for formatting
@app.template_filter('format_number')
def format_number(value):
    try:
        if pd.isna(value) or value is None:
            return "0"
        return "{:,}".format(int(float(value)))
    except (ValueError, TypeError):
        return "0"

@app.template_filter('format_float')
def format_float(value):
    try:
        if pd.isna(value) or value is None:
            return "0.0"
        return "{:,.1f}".format(float(value))
    except (ValueError, TypeError):
        return "0.0"

@app.template_filter('format_currency')
def format_currency(value):
    try:
        if pd.isna(value) or value is None:
            return "₦0.00"
        return "₦{:,.2f}".format(float(value))
    except (ValueError, TypeError):
        return "₦0.00"

def generate_chart_images_for_web(vehicle_distributions, road_results):
    """Generate base64-encoded chart images for web display."""
    chart_images = {}

    try:
        # Pie charts for each road
        for road_name, vehicles in vehicle_distributions.items():
            if vehicles and any(v['count'] > 0 for v in vehicles):
                labels = [v['vehicle_type'] for v in vehicles if v['count'] > 0]
                data = [v['count'] for v in vehicles if v['count'] > 0]
                chart_images[f"pie_{road_name.replace(' ', '_')}"] = generate_pie_chart(
                    data, labels, f"{road_name} Vehicle Distribution"
                ) or ""

        # Bar charts for comparative analysis
        road_names = list(road_results.keys())

        # People affected chart
        people_data = [road_results[r].get('total_people', 0) for r in road_names]
        if any(x > 0 for x in people_data):
            chart_images['people'] = generate_bar_chart(
                people_data, road_names, 'People Affected by Congestion', 'People'
            ) or ""

        # Fuel consumption chart
        fuel_data = [road_results[r].get('total_excess_fuel_l', 0) for r in road_names]
        if any(x > 0 for x in fuel_data):
            chart_images['fuel'] = generate_bar_chart(
                fuel_data, road_names, 'Excess Fuel Consumption', 'Liters'
            ) or ""

        # CO2 emissions chart
        co2_data = [road_results[r].get('total_co2_kg', 0) for r in road_names]
        if any(x > 0 for x in co2_data):
            chart_images['co2'] = generate_bar_chart(
                co2_data, road_names, 'Excess CO₂ Emissions', 'kg'
            ) or ""

        # Total cost chart
        cost_data = [(road_results[r].get('total_fuel_cost_naira', 0) +
                      road_results[r].get('total_productivity_loss_naira', 0)) for r in road_names]
        if any(x > 0 for x in cost_data):
            chart_images['cost'] = generate_bar_chart(
                cost_data, road_names, 'Total Cost (Fuel + Productivity Loss)', 'Cost', is_cost_chart=True
            ) or ""

    except Exception as e:
        logger.error(f"Error generating chart images for web: {str(e)}")
        flash(f"Error generating charts: {str(e)}", "error")

    return chart_images

def generate_pie_chart(data, labels, title):
    """Generate a base64-encoded pie chart for web display."""
    try:
        data = [float(x) if not pd.isna(x) else 0 for x in data]
        filtered_data = [d for d in data if d > 0]
        filtered_labels = [l for d, l in zip(data, labels) if d > 0]

        if not filtered_data:
            logger.warning(f"No valid data for pie chart: {title}")
            return None

        plt.figure(figsize=(8, 8))
        colors = ['#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56', '#9966FF', '#FF4500', '#2E8B57']
        plt.pie(filtered_data, labels=filtered_labels, autopct='%1.1f%%', colors=colors[:len(filtered_data)], startangle=90)
        plt.title(title, fontsize=14, pad=20)
        plt.axis('equal')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        logger.info(f"Generated pie chart: {title}")
        return f"data:image/png;base64,{img_data}"
    except Exception as e:
        logger.error(f"Error generating pie chart {title}: {str(e)}")
        return None

def generate_bar_chart(data, labels, title, ylabel, is_cost_chart=False):
    """Generate a base64-encoded bar chart for web display."""
    try:
        data = [float(x) if not pd.isna(x) else 0 for x in data]
        if not data or all(x == 0 for x in data):
            logger.warning(f"No valid data for bar chart: {title}")
            return None

        plt.figure(figsize=(12, 6))
        colors = ['#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56', '#9966FF']
        bars = plt.bar(range(len(labels)), data, color=colors[:len(labels)])
        plt.title(title, fontsize=16, pad=20)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')

        if is_cost_chart:
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₦{x:,.0f}'))
        else:
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2., height + max(data) * 0.01,
                         f'{height:,.0f}' if not is_cost_chart else f'₦{height:,.0f}',
                         ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        logger.info(f"Generated bar chart: {title}")
        return f"data:image/png;base64,{img_data}"
    except Exception as e:
        logger.error(f"Error generating bar chart {title}: {str(e)}")
        return None

def get_analysis_results(emission_model=EmissionModelType.BASIC):
    """Perform traffic analysis and generate results."""
    csv_file_path = "traffic_data.csv"
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')

    try:
        if os.path.exists(csv_file_path):
            logger.info(f"Loading traffic data from {csv_file_path}")
            model = TrafficAnalysisModel(csv_file_path=csv_file_path, emission_model=emission_model)
        else:
            logger.warning(f"CSV file {csv_file_path} not found. Using hardcoded data.")
            model = TrafficAnalysisModel(emission_model=emission_model)
    except Exception as e:
        logger.error(f"Error loading data: {e}. Using hardcoded data.")
        model = TrafficAnalysisModel(emission_model=emission_model)

    logger.info(f"Running traffic analysis with {emission_model.value} model...")
    results = model.analyze_all_roads()
    logger.debug(f"Analysis completed with {len(results['road_results'])} roads")

    report_df = model.generate_report()
    report_df = report_df.fillna(0)

    for col in ['Vehicle Count', 'People Affected']:
        if col in report_df.columns:
            report_df[col] = report_df[col].astype(int)

    report_df.to_csv('abuja_traffic_analysis_report.csv', index=False)
    logger.info("Generated report saved to abuja_traffic_analysis_report.csv")

    vehicle_distributions = {}
    for road_name, vehicle_data in model.road_data.items():
        vehicle_distributions[road_name] = [
            {"vehicle_type": vt, "count": int(count) if not pd.isna(count) else 0}
            for vt, count in vehicle_data.items() if count > 0
        ]

    chart_images = generate_chart_images_for_web(vehicle_distributions, results['road_results'])

    return results, report_df, vehicle_distributions, chart_images, emission_model.value

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

            if congested_travel_time <= free_flow_time:
                flash("Congested travel time must be greater than free flow time", "error")
                return redirect(url_for('data_entry'))

            if any(x <= 0 for x in [distance_km, free_flow_time, value_per_minute,
                                    fuel_cost_per_liter_diesel, fuel_cost_per_liter_petrol,
                                    emission_factor_petrol, emission_factor_diesel,
                                    free_flow_speed, congested_speed, avg_acceleration,
                                    avg_deceleration, idle_time_percentage]):
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
                        "Emission_Model": emission_model
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
        emission_model = EmissionModelType(emission_model_str)
        results, report_df, vehicle_distributions, chart_images, model_used = get_analysis_results(emission_model)
        summary = results['total_summary']
        road_results = results['road_results']
        model = TrafficAnalysisModel()
        emission_models = model.get_available_models()

        return render_template(
            'analysis.html',
            summary=summary,
            road_results=road_results,
            report_data=report_df.to_dict('records'),
            vehicle_distributions=vehicle_distributions,
            chart_images=chart_images,
            emission_models=emission_models,
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
        emission_model = EmissionModelType(emission_model_str)
        results, report_df, vehicle_distributions, chart_images, model_used = get_analysis_results(emission_model)
        summary = results['total_summary']
        road_results = results['road_results']
        model = TrafficAnalysisModel()
        emission_models = model.get_available_models()

        return render_template(
            'traffic_dashboard_pdf.html',
            summary=summary,
            road_results=road_results,
            report_data=report_df.to_dict('records'),
            vehicle_distributions=vehicle_distributions,
            chart_images=chart_images,
            emission_models=emission_models,
            current_emission_model=model_used,
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
        emission_model = EmissionModelType(emission_model_str)
        results, report_df, vehicle_distributions, _, model_used = get_analysis_results(emission_model)
        summary = results['total_summary']
        road_results = results['road_results']

        chart_files = save_charts_for_pdf(road_results, vehicle_distributions)
        absolute_chart_files = {key: os.path.abspath(path) for key, path in chart_files.items()}

        rendered = render_template(
            'pdf_report.html',
            summary=summary,
            road_results=road_results,
            report_data=report_df.to_dict('records'),
            vehicle_distributions=vehicle_distributions,
            chart_files=absolute_chart_files,
            emission_model=model_used,
            current_year=datetime.now().year,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M')
        )

        pdf_output = 'abuja_traffic_analysis_report.pdf'
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

        pdfkit.from_string(rendered, pdf_output, configuration=config, options=options)

        if not os.path.exists(pdf_output) or os.path.getsize(pdf_output) == 0:
            logger.error("PDF file was not created or is empty")
            flash("PDF file was not created or is empty. Check wkhtmltopdf and file permissions.", "error")
            return Response("PDF generation failed: File not created or empty", status=500)

        response = send_file(pdf_output, as_attachment=True, download_name='abuja_traffic_analysis_report.pdf')
        flash("PDF report generated successfully!", "success")

        # Clean up temporary chart files
        temp_dir = "temp_charts"
        for file in os.listdir(temp_dir):
            if file.endswith('.png'):
                try:
                    os.remove(os.path.join(temp_dir, file))
                    logger.debug(f"Deleted temporary file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {file}: {str(e)}")

        return response
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