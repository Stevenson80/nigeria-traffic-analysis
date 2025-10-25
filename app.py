#!/usr/bin/env python3
"""
Flask application for Nigeria Traffic Analysis System.
Handles web interface, data entry, analysis, and report generation.
"""
import os
import sys
import tempfile
from weasyprint import HTML
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
from flask import Flask, render_template, send_file, request, redirect, url_for, flash, jsonify, session
import pandas as pd
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
from reportlab.lib.utils import ImageReader
from traffic_analysis_models import TrafficAnalysisModel, EmissionModelType, CalculationMetric, valid_vehicle_types
import uuid

app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'nigeria-traffic-analysis-secret-key')

# Configure logging for Unicode support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Use stdout for better Unicode handling
        logging.FileHandler('app.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def fix_pdf_encoding():
    """Fix encoding issues for PDF generation"""
    import matplotlib.pyplot as plt
    # Configure matplotlib for proper PDF output
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    # Also ensure proper string encoding
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Apply PDF encoding fix on startup
fix_pdf_encoding()

# Constants
vehicle_classes = [
    {"id": "class1", "name": "Motorcycles", "key": "Class_1_Motorcycles"},
    {"id": "class2", "name": "Cars/SUVs", "key": "Class_2_Cars_SUVs"},
    {"id": "class3", "name": "Coasters/Buses", "key": "Class_3_Coasters_Buses"},
    {"id": "class4", "name": "Trucks", "key": "Class_4_Trucks"},
    {"id": "class5", "name": "Tankers/Trailers", "key": "Class_5_Tankers_Trailers"}
]
vehicle_class_list = [v["name"] for v in vehicle_classes]
valid_cities_by_state = {
    'Oyo': ['Ibadan', 'Ogbomoso', 'Oyo', 'Akobo'],
    'Kwara': ['Ilorin', 'Offa', 'Jebba'],
    'Lagos': ['Ikeja', 'Lagos Island', 'Badagry'],
    'Kano': ['Kano', 'Dala', 'Fagge'],
    'Rivers': ['Port Harcourt', 'Obio-Akpor', 'Bonny'],
    'Abuja': ['Garki', 'Wuse', 'Maitama', 'Asokoro']
}
NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River", "Delta",
    "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi",
    "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto",
    "Taraba", "Yobe", "Zamfara", "Federal Capital Territory"
]
NIGERIAN_CITIES = ['Abuja', 'Lagos', 'Kano', 'Ibadan', 'Port Harcourt', 'Benin City', 'Enugu', 'Kaduna', 'Ilorin',
                   'Jos', 'Owerri']
max_cities = 10
emission_models = [
    {"value": model.value, "name": model.display_name}
    for model in EmissionModelType
]


def generate_pdf_with_weasyprint(html_content, filename=None):
    """Generate PDF using WeasyPrint for better Unicode support"""
    if filename is None:
        filename = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name

    HTML(string=html_content).write_pdf(filename)
    return filename


def register_custom_fonts():
    """Register fonts that support Unicode characters"""
    try:
        # Try to register Arial Unicode MS or other Unicode fonts
        font_paths = [
            '/usr/share/fonts/truetype/msttcorefonts/Arial_Unicode_MS.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/System/Library/Fonts/Arial.ttf',
            'C:/Windows/Fonts/arial.ttf'
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('ArialUnicode', font_path))
                return 'ArialUnicode'

        # Fallback to built-in font
        pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
        return 'Arial'
    except:
        return 'Helvetica'


def generate_traffic_report_pdf(analysis_data, cities_data, filename=None, metric_enum=None):
    """Generate PDF report with proper Unicode support using WeasyPrint"""
    # Apply PDF encoding fix before generation
    fix_pdf_encoding()

    # Generate HTML content for WeasyPrint
    html_content = generate_pdf_html_content(analysis_data, cities_data, metric_enum)

    # Generate PDF using WeasyPrint
    if filename is None:
        filename = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name

    HTML(string=html_content).write_pdf(filename)
    return filename


def generate_pdf_html_content(analysis_data, cities_data, metric_enum=None):
    """Generate HTML content for PDF conversion with dynamic metric display"""

    # Determine which metric to display
    if metric_enum is None:
        # Try to get metric from analysis_data or default to CO2
        metric_enum = analysis_data.get('metric_enum', CalculationMetric.CO2_EMISSIONS)

    # Map metric to display names and units
    metric_info = {
        CalculationMetric.PRODUCTIVITY_LOSS: {
            'name': 'Productivity Loss',
            'unit': 'NGN',
            'css_class': 'naira',
            'total_label': 'Total Productivity Loss'
        },
        CalculationMetric.EXCESS_FUEL: {
            'name': 'Excess Fuel Used',
            'unit': 'NGN',
            'css_class': 'naira',
            'total_label': 'Total Excess Fuel Cost'
        },
        CalculationMetric.CO2_EMISSIONS: {
            'name': 'CO₂ Emissions',
            'unit': '',
            'css_class': 'co2',
            'total_label': 'Total CO₂ Emissions'
        }
    }

    current_metric = metric_info.get(metric_enum, metric_info[CalculationMetric.CO2_EMISSIONS])

    # Prepare chart images if available
    chart_images_html = ""
    if 'chart_images' in analysis_data:
        for chart_name, chart_path in analysis_data['chart_images'].items():
            if os.path.exists(chart_path):
                with open(chart_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    chart_images_html += f'''
                    <div class="chart-container">
                        <h3>{chart_name.replace('_', ' ').title()}</h3>
                        <img src="data:image/png;base64,{img_data}" alt="{chart_name}" style="max-width: 100%; height: auto;">
                    </div>
                    '''

    # Generate cities table with dynamic metric
    cities_table_html = ""
    for city_data in cities_data:
        city_name = city_data.get('city', 'Unknown')

        # Get the metric value for this city
        if metric_enum == CalculationMetric.PRODUCTIVITY_LOSS:
            metric_value = city_data.get('total_productivity_loss', 0)
            formatted_value = f"NGN{metric_value:,.0f}" if metric_value else "NGN0"
        elif metric_enum == CalculationMetric.EXCESS_FUEL:
            metric_value = city_data.get('total_excess_fuel', 0)
            formatted_value = f"NGN{metric_value:,.0f}" if metric_value else "NGN0"
        else:  # CO2_EMISSIONS
            metric_value = city_data.get('total_co2_emissions', 0)
            formatted_value = f"{metric_value:,.0f} kg" if metric_value else "0 kg"

        cities_table_html += f'''
        <div class="city-section">
            <h3>{city_name}</h3>
            <table class="city-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Vehicles</td>
                    <td>{city_data.get('total_vehicles', 0):,.1f}</td>
                </tr>
                <tr>
                    <td>People Affected</td>
                    <td>{city_data.get('total_people', 0):,.1f}</td>
                </tr>
                <tr>
                    <td>{current_metric['name']}</td>
                    <td class="{current_metric['css_class']}">{formatted_value}</td>
                </tr>
            </table>
        </div>
        '''

    # Generate detailed vehicle breakdown table with dynamic metric
    detailed_table_html = ""
    detailed_headers = ['State', 'City', 'Vehicle Type', 'Vehicle Count', 'Occupancy', f'{current_metric["name"]}']
    detailed_data = []

    for city_data in cities_data:
        for vehicle_type, data in city_data.get('vehicle_details', {}).items():
            # Get the appropriate metric value for this vehicle type
            if metric_enum == CalculationMetric.PRODUCTIVITY_LOSS:
                metric_value = data.get('productivity_loss', 0)
                formatted_metric = f"NGN{metric_value:,.1f}" if metric_value else "NGN0.0"
            elif metric_enum == CalculationMetric.EXCESS_FUEL:
                metric_value = data.get('excess_fuel', 0)
                formatted_metric = f"NGN{metric_value:,.1f}" if metric_value else "NGN0.0"
            else:  # CO2_EMISSIONS
                metric_value = data.get('co2_emissions', 0)
                formatted_metric = f"{metric_value:,.1f} kg" if metric_value else "0.0 kg"

            detailed_data.append([
                city_data.get('state', ''),
                city_data.get('city', ''),
                vehicle_type,
                f"{data.get('count', 0):.1f}",
                f"{data.get('occupancy', 0):.1f}",
                formatted_metric
            ])

    if detailed_data:
        detailed_table_html = f'''
        <div class="detailed-table">
            <h3>Detailed Vehicle Analysis</h3>
            <table class="vehicle-breakdown">
                <thead>
                    <tr>
                        <th>State</th>
                        <th>City</th>
                        <th>Vehicle Type</th>
                        <th>Vehicle Count</th>
                        <th>Occupancy</th>
                        <th>{current_metric['name']}</th>
                    </tr>
                </thead>
                <tbody>
        '''
        for row in detailed_data:
            detailed_table_html += f'''
                    <tr>
                        <td>{row[0]}</td>
                        <td>{row[1]}</td>
                        <td>{row[2]}</td>
                        <td>{row[3]}</td>
                        <td>{row[4]}</td>
                        <td>{row[5]}</td>
                    </tr>
            '''
        detailed_table_html += '''
                </tbody>
            </table>
        </div>
        '''

    # Get total metric value for summary
    if metric_enum == CalculationMetric.PRODUCTIVITY_LOSS:
        total_metric = analysis_data.get('total_productivity_loss', 0)
        formatted_total = f"NGN{total_metric:,.0f}" if total_metric else "NGN0"
    elif metric_enum == CalculationMetric.EXCESS_FUEL:
        total_metric = analysis_data.get('total_excess_fuel', 0)
        formatted_total = f"NGN{total_metric:,.0f}" if total_metric else "NGN0"
    else:  # CO2_EMISSIONS
        total_metric = analysis_data.get('total_co2_emissions', 0)
        formatted_total = f"{total_metric:,.0f} kg" if total_metric else "0 kg"

    # GET USER INPUTS - these should always be available
    user_inputs = analysis_data.get('user_inputs', {})

    if not user_inputs:
        logger.error("No user inputs found for PDF generation!")
        # Instead of falling back to wrong data, use placeholder
        user_inputs = {
            'fuel_cost_petrol': 900.0,  # Use numbers, not strings
            'fuel_cost_diesel': 1220.0,
            'value_per_minute': 15.67,
            'distance_km': 8.0,
            'free_flow_time': 10.0,
            'congested_travel_time': 35.0,
            'analysis_period': 1.3,
            'flow_rate': 1500,
            'emission_factor_petrol': 2.31,
            'emission_factor_diesel': 2.68,
            'road_name': 'Unknown'
        }

    # ALWAYS use user inputs - no fallbacks to processed data
    # Convert to float to ensure proper formatting
    fuel_cost_petrol = float(user_inputs.get('fuel_cost_petrol', 900.0))
    fuel_cost_diesel = float(user_inputs.get('fuel_cost_diesel', 1220.0))
    value_per_minute = float(user_inputs.get('value_per_minute', 15.67))
    distance_km = float(user_inputs.get('distance_km', 8.0))
    free_flow_time = float(user_inputs.get('free_flow_time', 10.0))
    congested_travel_time = float(user_inputs.get('congested_travel_time', 35.0))
    analysis_period = float(user_inputs.get('analysis_period', 1.3))
    flow_rate = float(user_inputs.get('flow_rate', 1500))
    emission_factor_petrol = float(user_inputs.get('emission_factor_petrol', 2.31))
    emission_factor_diesel = float(user_inputs.get('emission_factor_diesel', 2.68))
    road_name = user_inputs.get('road_name', 'Unknown')

    # Log what we're using
    logger.info(f"PDF using USER INPUTS - Times: {free_flow_time}/{congested_travel_time}, "
                f"Period: {analysis_period}, Fuel: {fuel_cost_petrol}/{fuel_cost_diesel}")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 0;
                padding: 20px;
                color: #333;
                line-height: 1.6;
            }}
            .header {{ 
                text-align: center; 
                border-bottom: 3px solid #2c3e50;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .title {{ 
                font-size: 24px; 
                font-weight: bold; 
                color: #2c3e50;
                margin-bottom: 10px;
            }}
            .subtitle {{ 
                font-size: 18px; 
                color: #34495e;
                margin-bottom: 15px;
            }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: #f8f9fa;
            }}
            .summary-table th {{
                background-color: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }}
            .summary-table td {{
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }}
            .summary-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .city-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            .city-table th {{
                background-color: #2c3e50;
                color: white;
                padding: 10px;
                text-align: left;
            }}
            .city-table td {{
                padding: 10px;
                border: 1px solid #ddd;
            }}
            .vehicle-breakdown {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 12px;
            }}
            .vehicle-breakdown th {{
                background-color: #34495e;
                color: white;
                padding: 8px;
                text-align: center;
            }}
            .vehicle-breakdown td {{
                padding: 8px;
                border: 1px solid #ddd;
                text-align: center;
            }}
            .vehicle-breakdown tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .city-section {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }}
            .chart-container {{
                margin: 25px 0;
                text-align: center;
            }}
            .chart-container img {{
                max-width: 90%;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: white;
            }}
            .methodology {{
                background-color: #e8f4f8;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .conclusions {{
                background-color: #f0f8f0;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
                font-size: 12px;
            }}
            .co2 {{ color: #e74c3c; font-weight: bold; }}
            .naira {{ color: #27ae60; font-weight: bold; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            h1 {{ font-size: 28px; }}
            h2 {{ font-size: 22px; border-bottom: 2px solid #bdc3c7; padding-bottom: 10px; }}
            h3 {{ font-size: 18px; color: #34495e; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 8px; }}
            .highlight {{ background-color: #fffacd; padding: 2px 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">Nigeria Traffic Analysis System</div>
            <div class="subtitle">{current_metric['name']} Analysis Report</div>
            <div><strong>Generated on {analysis_data.get('generation_date', 'N/A')} at {analysis_data.get('generation_time', 'N/A')}</strong></div>
            <div>Emission Model: <strong>{analysis_data.get('emission_model', 'N/A')}</strong></div>
            <div>Road: <strong>{road_name}</strong></div>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <table class="summary-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Vehicles Analyzed</td>
                    <td>{analysis_data.get('total_vehicles', 0):,.1f}</td>
                </tr>
                <tr>
                    <td>Total People Affected</td>
                    <td>{analysis_data.get('total_people', 0):,.1f}</td>
                </tr>
                <tr>
                    <td>{current_metric['total_label']}</td>
                    <td class="{current_metric['css_class']}">{formatted_total}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Features</h2>
            <ul>
                <li>Accurate vehicle counting and classification</li>
                <li>Real-time congestion analysis</li>
                <li>Emission modeling ({analysis_data.get('emission_model', 'N/A')})</li>
                <li>{current_metric['name']} assessment</li>
            </ul>
        </div>

        <div class="methodology">
            <h2>Methodology</h2>
            <p>This assessment uses vehicle-specific parameters to calculate {current_metric['name'].lower()}:</p>
            <ul>
                <li>Fuel Prices: <span class="naira">NGN{fuel_cost_petrol:.2f}/L</span> (petrol), <span class="naira">NGN{fuel_cost_diesel:.2f}/L</span> (diesel)</li>
                <li>Emission Factors: <span class="highlight">{emission_factor_petrol:.2f} kg CO₂/L</span> (petrol), <span class="highlight">{emission_factor_diesel:.2f} kg CO₂/L</span> (diesel)</li>
                <li>Productivity Value: <span class="naira">NGN{value_per_minute:.2f}/minute</span></li>
                <li>Corridor Length: <span class="highlight">{distance_km:.1f} km</span></li>
                <li>Time Parameters: Free flow: <span class="highlight">{free_flow_time:.1f} min</span>, Congested: <span class="highlight">{congested_travel_time:.1f} min</span></li>
                <li>Analysis Period: <span class="highlight">{analysis_period:.1f} hours</span></li>
                <li>Flow Rate: <span class="highlight">{flow_rate:.0f} vehicles/hour</span></li>
                <li>Road Name: <span class="highlight">{road_name}</span></li>
            </ul>
        </div>

        <div class="section">
            <h2>Per-City Analysis</h2>
            {cities_table_html}
        </div>

        {detailed_table_html}

        {chart_images_html}

        <div class="conclusions">
            <h2>Conclusions & Recommendations</h2>
            <h3>Key Findings</h3>
            <ul>
                <li>Analysis shows significant {current_metric['name'].lower()} from traffic congestion</li>
                <li>{current_metric['total_label']}: <span class="{current_metric['css_class']}">{formatted_total}</span></li>
                <li>Affected commuters: <span class="highlight">{analysis_data.get('total_people', 0):,.0f} people</span></li>
                <li>Analysis performed using {analysis_data.get('emission_model', 'N/A')} emission model</li>
            </ul>

            <h3>Strategic Recommendations</h3>
            <ul>
                <li>Implement targeted traffic management systems</li>
                <li>Enhance public transportation options</li>
                <li>Consider congestion pricing to reduce {current_metric['name'].lower()}</li>
                <li>Expand environmental monitoring</li>
            </ul>
        </div>

        <div class="footer">
            Generated on {analysis_data.get('generation_date', 'N/A')} at {analysis_data.get('generation_time', 'N/A')} by Nigeria Traffic Analysis System
        </div>
    </body>
    </html>
    """

    return html_content


def validate_co2_calculation(model, form_data, metric_enum, emission_model):
    """Validate that CO₂ calculations use total fuel consumption."""
    if metric_enum == CalculationMetric.CO2_EMISSIONS:
        logger.info("CO₂ calculation validation - using TOTAL fuel consumption method")
        # Log for debugging
        for state, cities_data in form_data.items():
            for city, vehicle_data in cities_data.items():
                logger.debug(f"CO₂ calculation for {city}, {state} using {emission_model.value}")


def validate_traffic_data(data):
    """Validate traffic data inputs."""
    errors = []

    # Validate idle time percentage
    idle_time = data.get('idle_time_percentage', 20.0)
    if idle_time > 100 or idle_time < 0:
        errors.append("Idle time percentage must be between 0 and 100")
        # Auto-correct to reasonable value
        data['idle_time_percentage'] = max(0, min(100, idle_time))

    # Validate other advanced parameters
    if data.get('stops_per_km', 2.0) < 0:
        errors.append("Stops per km cannot be negative")

    # Validate fuel prices
    if data.get('fuel_cost_petrol', 0) <= 0:
        errors.append("Petrol fuel cost must be greater than 0")

    if data.get('fuel_cost_diesel', 0) <= 0:
        errors.append("Diesel fuel cost must be greater than 0")

    return errors, data


def validate_form_data(form_data):
    """Validate form data for traffic analysis."""
    errors = []

    # Validate state
    state = form_data.get('state', '').title()
    if not state or state not in NIGERIAN_STATES:
        errors.append("Please select a valid Nigerian state")

    # Validate cities
    cities = [city.strip().title() for city in form_data.getlist('cities[]')]
    if not cities:
        errors.append("Please select at least one city")
    elif len(cities) != len(set(cities)):
        errors.append("Duplicate city names are not allowed")

    # Validate region
    region = form_data.get('region', '').strip().title()
    if not region:
        errors.append("Region is required")

    # Validate numeric inputs
    numeric_fields = [
        'congested_travel_time', 'distance_km', 'free_flow_time', 'productivity_value',
        'fuel_cost_petrol', 'fuel_cost_diesel', 'emission_factor_petrol',
        'emission_factor_diesel', 'free_flow_speed', 'congested_speed', 'avg_acceleration',
        'avg_deceleration', 'idle_time_percentage', 'stops_per_km', 'road_grade', 'temperature_c'
    ]

    for field in numeric_fields:
        try:
            value = float(form_data.get(field, 0))
            if value < 0 and field not in ['road_grade', 'temperature_c']:  # Allow negative for grade and temp
                errors.append(f"{field.replace('_', ' ').title()} cannot be negative")
        except (ValueError, TypeError):
            errors.append(f"{field.replace('_', ' ').title()} must be a valid number")

    # Validate Barth coefficients
    barth_fields = ['barth_alpha', 'barth_beta', 'barth_gamma']
    for field in barth_fields:
        try:
            value = float(form_data.get(field, 0))
            if value <= 0.00001:
                errors.append(f"{field.replace('_', ' ').title()} must be greater than 0.00001")
        except (ValueError, TypeError):
            errors.append(f"{field.replace('_', ' ').title()} must be a valid number")

    # Validate travel time logic
    try:
        congested_time = float(form_data.get('congested_travel_time', 0))
        free_flow_time = float(form_data.get('free_flow_time', 0))
        if congested_time <= free_flow_time:
            errors.append("Congested travel time must be greater than free flow time")
    except (ValueError, TypeError):
        pass

    # Validate speed logic
    try:
        congested_speed = float(form_data.get('congested_speed', 0))
        free_flow_speed = float(form_data.get('free_flow_speed', 0))
        if congested_speed >= free_flow_speed:
            errors.append("Congested speed must be less than free flow speed")
    except (ValueError, TypeError):
        pass

    # Validate vehicle counts
    has_valid_counts = False
    for city in cities:
        normalized_city = city.lower().replace(' ', '_')
        for v in vehicle_classes:
            count_field = f'Real_Vehicle_Count_{normalized_city}_{v["key"]}'
            count_str = form_data.get(count_field, '')

            if count_str:
                try:
                    count = int(count_str)
                    if count < 0:
                        errors.append(f"Vehicle count for {v['name']} in {city} must be non-negative")
                    if count > 0:
                        has_valid_counts = True

                    # Validate occupancy rate if count > 0
                    vor_field = f'Real_VOR_{normalized_city}_{v["key"]}'
                    vor_str = form_data.get(vor_field, '')
                    if count > 0 and vor_str:
                        try:
                            vor = float(vor_str)
                            if vor <= 0:
                                errors.append(
                                    f"Occupancy rate for {v['name']} in {city} must be positive when count is non-zero")
                        except (ValueError, TypeError):
                            errors.append(f"Occupancy rate for {v['name']} in {city} must be a valid number")

                except (ValueError, TypeError):
                    errors.append(f"Vehicle count for {v['name']} in {city} must be an integer")

    if not has_valid_counts:
        errors.append("At least one non-zero vehicle count is required")

    return errors, {
        'state': state,
        'cities': cities,
        'region': region,
        'form_data': form_data
    }


# Utility functions
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
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


def format_metric_value(value, metric_enum):
    """Format values with proper units based on metric type."""
    try:
        if pd.isna(value) or value is None:
            value = 0
        value = float(value)

        if metric_enum == CalculationMetric.PRODUCTIVITY_LOSS:
            return f"NGN{value:,.0f}"
        elif metric_enum == CalculationMetric.EXCESS_FUEL:
            return f"NGN{value:,.0f}"
        elif metric_enum == CalculationMetric.CO2_EMISSIONS:
            return f"{value:,.0f} kg"
        else:
            return f"{value:,.0f}"
    except (ValueError, TypeError):
        return "0"


def is_valid_city(state, city):
    if not city or not city.strip():
        return False
    # Allow any city for all states - remove strict validation
    return True


def generate_fallback_pdf():
    """Generate a simple PDF when analysis data is not available."""
    try:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .error { color: #e74c3c; font-weight: bold; }
                .info { color: #3498db; }
            </style>
        </head>
        <body>
            <h1>Nigeria Traffic Analysis System</h1>
            <h2 class="error">PDF Report Generation Failed</h2>
            <p class="info">Please run a traffic analysis first from the Dashboard.</p>
            <p class="info">Then try downloading the PDF again.</p>
        </body>
        </html>
        """

        filename = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
        HTML(string=html_content).write_pdf(filename)

        return send_file(
            filename,
            as_attachment=True,
            download_name='traffic_analysis_error.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error in fallback PDF: {str(e)}")
        flash("Could not generate PDF report", "error")
        return redirect(url_for('dashboard_get'))


# Template filters
@app.template_filter('format_number')
def format_number(value):
    try:
        if pd.isna(value) or value is None:
            return "0"
        return "{:,}".format(int(float(value)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for number formatting: {value}")
        return "0"


@app.template_filter('format_float')
def format_float(value):
    try:
        if pd.isna(value) or value is None:
            return "0.0"
        return "{:,.1f}".format(float(value))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for float formatting: {value}")
        return "0.0"


@app.template_filter('format_currency')
def format_currency(value):
    try:
        if pd.isna(value) or value is None:
            return "NGN0.00"
        return "NGN{:,.2f}".format(float(value))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for currency formatting: {value}")
        return "NGN0.00"


@app.template_filter('get_metric_name')
def get_metric_name(metrics, metric):
    return {
        'productivity_loss': 'Productivity Loss',
        'excess_fuel': 'Excess Fuel Used',
        'co2_emissions': 'CO₂ Emissions'
    }.get(metric, 'Unknown')


@app.template_filter('get_metric_abbr')
def get_metric_abbr(metrics, metric):
    return {
        'productivity_loss': 'PL',
        'excess_fuel': 'EF',
        'co2_emissions': 'CO2'
    }.get(metric, 'Unknown')


@app.route('/get_cities', methods=['GET'])
def get_cities():
    return jsonify(valid_cities_by_state)


def get_analysis_results(metric, emission_model, form_data=None):
    """Perform traffic analysis using user input data only."""
    try:
        model = TrafficAnalysisModel(emission_model=emission_model)

        # Process form data if provided
        if form_data:
            logger.info(f"Processing form data with {len(form_data)} states")
            for state, cities_data in form_data.items():
                for city, vehicle_data in cities_data.items():
                    model.add_city_data(state, city, vehicle_data)
                    logger.info(f"Added data for {city}, {state}: {len(vehicle_data)} vehicle records")
        else:
            # If no form data, check session
            if 'analysis_data' in session:
                form_data = session['analysis_data'].get('form_data')
                if form_data:
                    logger.info(f"Using form data from session with {len(form_data)} states")
                    for state, cities_data in form_data.items():
                        for city, vehicle_data in cities_data.items():
                            model.add_city_data(state, city, vehicle_data)
            else:
                # If no form data and no session data, check if we have data from the model
                if not model.state_city_data:
                    logger.error("No analysis data available")
                    raise ValueError("No traffic data available for analysis")

        logger.info(f"Running traffic analysis with {emission_model.value} model for metric {metric}...")

        # Add CO2 calculation validation
        validate_co2_calculation(model, form_data, metric, emission_model)

        results = model.calculate_metric(metric=metric, model=emission_model)
        if not results or not results['city_results']:
            logger.error("Analysis returned no results")
            raise ValueError("Traffic analysis failed to produce results")

        report_df = model.generate_report(metric=metric, model=emission_model)
        report_df = report_df.fillna(0)

        vehicle_distributions = {}
        for state in model.state_city_data:
            for city in model.state_city_data[state]:
                distribution_key = f"{state}_{city}"
                distribution = model.get_vehicle_distribution(state, city)
                vehicle_distributions[distribution_key] = distribution
                logger.info(f"Vehicle distribution for {city}, {state}: {distribution}")

        chart_images = model.generate_chart_images(metric=metric)
        results = convert_numpy_types(results)

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
                    pass

        return results, report_data, vehicle_distributions, chart_images, emission_model.value, model
    except Exception as e:
        logger.error(f"Error in get_analysis_results: {e}", exc_info=True)
        raise


@app.route('/', methods=['GET'])
def welcome():
    """Render the welcome page with metric and formula selection."""
    metrics = [
        {'id': 'productivity_loss', 'name': 'Productivity Loss'},
        {'id': 'excess_fuel', 'name': 'Excess Fuel Used'},
        {'id': 'co2_emissions', 'name': 'CO₂ Emissions'}
    ]
    valid_metrics = {metric['id'] for metric in metrics}
    valid_formulas = {formula['value'] for formula in emission_models}

    default_formula = 'basic'

    stats = {'cities_analyzed': 0}  # No CSV data, always 0

    vehicle_descriptions = {
        'Motorcycles': 'Motorcycles',
        'Cars/SUVs': 'Cars, SUVs, Sedans, Hatchbacks',
        'Coasters/Buses': 'Coasters, Buses, Minibuses',
        'Trucks': 'Trucks, Pickups, Lorries',
        'Tankers/Trailers': 'Tankers, Trailers, Heavy Trailers'
    }

    return render_template(
        'welcome.html',
        metrics=metrics,
        formulas=emission_models,
        stats=stats,
        vehicle_class_list=vehicle_class_list,
        vehicle_descriptions=vehicle_descriptions,
        current_year=datetime.now().year,
        default_metric='productivity_loss',
        default_formula=default_formula
    )


@app.route('/data_entry', methods=['GET', 'POST'])
def data_entry():
    """Handle data entry form and save traffic data."""
    # Get parameters from URL - these should take priority
    metric = request.args.get('metric', '')
    emission_model_str = request.args.get('emission_model', '')
    state = request.args.get('state', '')
    region = request.args.get('region', '')

    # Only use session values if URL parameters are empty
    if not metric:
        metric = session.get('metric', 'productivity_loss')
    if not emission_model_str:
        emission_model_str = session.get('emission_model', 'basic')
    if not state:
        state = session.get('state', '')
    if not region:
        region = session.get('region', '')

    # Get cities from URL or session
    cities = request.args.getlist('cities') or session.get('cities', [])

    if not state:
        state = list(valid_cities_by_state.keys())[0] if valid_cities_by_state else 'Unknown'

    try:
        emission_model = EmissionModelType(emission_model_str)
    except ValueError:
        logger.warning(f"Invalid emission model: {emission_model_str}. Defaulting to BASIC.")
        emission_model = EmissionModelType.BASIC
    current_emission_model = emission_model.value

    if request.method == 'POST':
        state = request.form.get('state', '').title()
        region = request.form.get('region', '').strip().title()
        cities = [city.strip().title() for city in request.form.getlist('cities[]')]

        if not state or not cities or not region:
            flash('Please select a state, region, and at least one city.', 'error')
            return redirect(url_for('data_entry', metric=metric, emission_model=emission_model_str))

        if not all(is_valid_city(state, city) for city in cities):
            flash(f'Invalid cities selected for state {state}.', 'error')
            return redirect(url_for('data_entry', metric=metric, emission_model=emission_model_str))

        session['metric'] = metric
        session['emission_model'] = emission_model_str
        session['state'] = state
        session['cities'] = cities
        session['region'] = region

    model = TrafficAnalysisModel(emission_model=emission_model)
    vehicle_parameters = model.vehicle_parameters
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')

    # Use empty or sensible starting values instead of model defaults:
    current_data = {
        'entry_date': current_date,
        'entry_time': current_time,
        'congested_travel_time': '',  # Let user fill this
        'distance_km': '',  # Let user fill this
        'free_flow_time': '',  # Let user fill this
        'productivity_value': '',  # Let user fill this
        'fuel_cost_petrol': '',  # Let user fill this
        'fuel_cost_diesel': '',  # Let user fill this
        'emission_factor_petrol': '2.31',  # Only provide defaults for truly advanced technical fields
        'emission_factor_diesel': '2.68',  # Only provide defaults for truly advanced technical fields
        'free_flow_speed': '',  # Let user fill this
        'congested_speed': '',  # Let user fill this
        'avg_acceleration': '',  # Let user fill this
        'avg_deceleration': '',  # Let user fill this
        'idle_time_percentage': '',  # Let user fill this
        'stops_per_km': '',  # Let user fill this
        'road_grade': '',  # Let user fill this
        'temperature_c': '',  # Let user fill this
        'emission_model': emission_model.value,
        'barth_alpha': '0.00015',  # Only provide defaults for truly advanced technical fields
        'barth_beta': '0.0008',  # Only provide defaults for truly advanced technical fields
        'barth_gamma': '0.0005',  # Only provide defaults for truly advanced technical fields
        'barth_traffic_flow': '0.001',  # Only provide defaults for truly advanced technical fields
        'barth_road_gradient': '0.0005',  # Only provide defaults for truly advanced technical fields
        'barth_acceleration': '0.0003',  # Only provide defaults for truly advanced technical fields
        'road': ''  # Let user fill this
    }
    for v in vehicle_classes:
        field_name = f"emission_factor_{v['key']}"
        current_data[field_name] = str(
            model.vehicle_parameters.get(v['name'], {}).get('emission_factor', 0.2)
        )

    # No CSV loading - start with empty city_data
    city_data = {}

    logger.debug(
        f"Rendering data_entry.html with state={state}, cities={cities}, metric={metric}, emission_model={current_emission_model}, region={region}")

    return render_template(
        'data_entry.html',
        max_cities=max_cities,
        num_current_cities=len(cities),
        cities=cities,
        vehicleClasses=vehicle_classes,
        vehicle_class_list=vehicle_class_list,
        current_data=current_data,
        emission_models=emission_models,
        current_emission_model=current_emission_model,
        metric=metric,  # Pass the actual metric value
        emission_model=emission_model.value,  # Pass the actual emission model value
        metric_display={
            'productivity_loss': 'Productivity Loss',
            'excess_fuel': 'Excess Fuel Used',
            'co2_emissions': 'CO₂ Emissions'
        }.get(metric, 'Productivity Loss'),
        current_year=datetime.now().year,
        nigerian_cities=NIGERIAN_CITIES,
        valid_cities_by_state=valid_cities_by_state,
        state=state,  # Pass the actual state value
        city_data=city_data,
        vehicle_classes=vehicle_classes,
        vehicle_parameters=vehicle_parameters,
        region=region  # Pass the actual region value
    )


@app.route('/dashboard', methods=['POST'])
def dashboard():
    """Process data entry form submission and render dashboard."""
    try:
        # Get currency from hidden input
        currency = request.form.get('currency', 'NGN')
        logger.info(f"Using currency: {currency}")

        emission_model_form = request.form.get('emission_model', 'basic')
        try:
            emission_model = EmissionModelType(emission_model_form)
        except ValueError:
            logger.warning(f"Invalid emission model: {emission_model_form}. Defaulting to BASIC.")
            emission_model = EmissionModelType.BASIC

        logger.info(f"=== DASHBOARD: Using emission model: {emission_model.value} ===")

        # Validate form data first
        validation_errors, validated_data = validate_form_data(request.form)
        if validation_errors:
            for error in validation_errors:
                flash(error, "error")
            return redirect(url_for('data_entry'))

        state = validated_data['state']
        cities = validated_data['cities']
        region = validated_data['region']
        form_data = validated_data['form_data']

        model = TrafficAnalysisModel(emission_model=emission_model)

        entry_date = request.form.get('entry_date', datetime.now().strftime('%Y-%m-%d'))
        entry_time = request.form.get('entry_time', datetime.now().strftime('%H:%M'))

        # Calculate actual vehicle distribution from user input counts
        def calculate_actual_distribution(cities, vehicle_classes, form_data):
            """Calculate distribution percentages from actual vehicle counts entered by user"""
            total_counts = {}

            for city in cities:
                normalized_city = city.lower().replace(' ', '_')
                for v in vehicle_classes:
                    count_field = f'Real_Vehicle_Count_{normalized_city}_{v["key"]}'
                    count_str = form_data.get(count_field, '0')
                    count = int(count_str) if count_str.strip() else 0

                    if count > 0:
                        total_counts[v['name']] = total_counts.get(v['name'], 0) + count

            # Calculate percentages
            total_vehicles = sum(total_counts.values())
            if total_vehicles == 0:
                return {v['name']: 0 for v in vehicle_classes}

            distribution = {}
            for vehicle_type, count in total_counts.items():
                distribution[vehicle_type] = round((count / total_vehicles) * 100, 2)

            # Fill in zeros for vehicle types with no counts
            for v in vehicle_classes:
                if v['name'] not in distribution:
                    distribution[v['name']] = 0

            return distribution

        # Calculate actual distribution from user input
        actual_distribution = calculate_actual_distribution(cities, vehicle_classes, request.form)
        logger.info(f"Actual vehicle distribution calculated from user counts: {actual_distribution}")

        # Use the actual distribution for calculations
        class_distributions = actual_distribution

        # Use validated numeric values with UPDATED default fuel prices
        try:
            # CRITICAL: Extract ALL parameters exactly as entered in the form
            congested_travel_time = float(request.form.get('congested_travel_time', 35.0))
            distance_km = float(request.form.get('distance_km', 8.0))
            free_flow_time = float(request.form.get('free_flow_time', 10.0))
            productivity_value = float(request.form.get('productivity_value', 15.67))
            # Use EXACT fuel prices from form, not model defaults
            fuel_cost_petrol = float(request.form.get('fuel_cost_petrol', 900.0))
            fuel_cost_diesel = float(request.form.get('fuel_cost_diesel', 1220.0))
            emission_factor_petrol = float(request.form.get('emission_factor_petrol', 2.31))
            emission_factor_diesel = float(request.form.get('emission_factor_diesel', 2.68))
            free_flow_speed = float(request.form.get('free_flow_speed', 60.0))
            congested_speed = float(request.form.get('congested_speed', 8.0))
            avg_acceleration = float(request.form.get('avg_acceleration', 0.5))
            avg_deceleration = float(request.form.get('avg_deceleration', 0.5))
            idle_time_percentage = float(request.form.get('idle_time_percentage', 20.0))
            stops_per_km = float(request.form.get('stops_per_km', 2.0))
            road_grade = float(request.form.get('road_grade', 0.0))
            temperature_c = float(request.form.get('temperature_c', 25.0))
            barth_alpha = float(request.form.get('barth_alpha', 0.00015))
            barth_beta = float(request.form.get('barth_beta', 0.0008))
            barth_gamma = float(request.form.get('barth_gamma', 0.0005))
            barth_traffic_flow = float(request.form.get('barth_traffic_flow', 0.001))
            barth_road_gradient = float(request.form.get('barth_road_gradient', 0.0005))
            barth_acceleration = float(request.form.get('barth_acceleration', 0.0003))
            flow_rate = float(request.form.get('flow_rate', 1500))
            analysis_period = float(request.form.get('analysis_period', 1.3))
            road_name = request.form.get('road', 'Unknown').strip().title() or 'Unknown'
        except ValueError as e:
            flash(f"Invalid numeric input: {str(e)}", "error")
            return redirect(url_for('data_entry'))

        # Additional validation for traffic data
        traffic_data = {
            'idle_time_percentage': idle_time_percentage,  # Already in percentage
            'stops_per_km': stops_per_km,
            'fuel_cost_petrol': fuel_cost_petrol,
            'fuel_cost_diesel': fuel_cost_diesel
        }
        traffic_errors, corrected_data = validate_traffic_data(traffic_data)
        if traffic_errors:
            for error in traffic_errors:
                flash(error, "warning")
            # Use corrected values
            idle_time_percentage = corrected_data['idle_time_percentage']

        # === DEBUG: Checking user inputs ===
        logger.info("=== DEBUG: Checking user inputs ===")
        logger.info(f"User input - Analysis Period: {analysis_period}")
        logger.info(f"User input - Flow Rate: {flow_rate}")
        logger.info(f"User input - Congested Time: {congested_travel_time}")
        logger.info(f"User input - Free Flow Time: {free_flow_time}")
        logger.info(f"User input - Distance: {distance_km}")
        logger.info(f"User input - Actual Vehicle Distribution: {actual_distribution}")

        # === STORE RAW USER INPUTS FOR PDF GENERATION ===
        raw_user_inputs = {
            'congested_travel_time': congested_travel_time,
            'free_flow_time': free_flow_time,
            'distance_km': distance_km,
            'value_per_minute': productivity_value,
            'fuel_cost_petrol': fuel_cost_petrol,
            'fuel_cost_diesel': fuel_cost_diesel,
            'analysis_period': analysis_period,
            'flow_rate': flow_rate,
            'currency': currency,
            # Store RAW vehicle counts
            'vehicle_counts': {},
            'actual_distribution': actual_distribution,
            # ADD THESE CRITICAL PARAMETERS:
            'emission_factor_petrol': emission_factor_petrol,
            'emission_factor_diesel': emission_factor_diesel,
            'road_name': road_name,
            # Store ALL advanced parameters
            'free_flow_speed': free_flow_speed,
            'congested_speed': congested_speed,
            'avg_acceleration': avg_acceleration,
            'avg_deceleration': avg_deceleration,
            'idle_time_percentage': idle_time_percentage,
            'stops_per_km': stops_per_km,
            'road_grade': road_grade,
            'temperature_c': temperature_c,
            'barth_alpha': barth_alpha,
            'barth_beta': barth_beta,
            'barth_gamma': barth_gamma,
            'barth_traffic_flow': barth_traffic_flow,
            'barth_road_gradient': barth_road_gradient,
            'barth_acceleration': barth_acceleration
        }

        # Store raw vehicle counts for each city
        for city in cities:
            normalized_city = city.lower().replace(' ', '_')
            raw_user_inputs['vehicle_counts'][city] = {}

            for v in vehicle_classes:
                count_field = f'Real_Vehicle_Count_{normalized_city}_{v["key"]}'
                vor_field = f'Real_VOR_{normalized_city}_{v["key"]}'

                count_str = request.form.get(count_field, '0')
                vor_str = request.form.get(vor_field, '')

                raw_count = int(count_str) if count_str.strip() else 0
                raw_vor = float(vor_str) if vor_str.strip() else model.vehicle_parameters.get(v['name'], {}).get(
                    'occupancy_avg', 1.0)

                raw_user_inputs['vehicle_counts'][city][v['name']] = {
                    'count': raw_count,
                    'occupancy': raw_vor
                }

        logger.info(f"Stored RAW user inputs: {raw_user_inputs}")

        # === DEBUG: Log actual parameters being used ===
        logger.info("=== ACTUAL PARAMETERS BEING USED ===")
        # Fix the vehicle count calculation
        total_vehicles = 0
        for city_data in raw_user_inputs['vehicle_counts'].values():
            for vehicle_data in city_data.values():
                total_vehicles += vehicle_data['count']
        logger.info(f"Vehicle counts: {total_vehicles}")

        # Fixed Naira symbol in logs
        logger.info(f"Fuel - Petrol: NGN{fuel_cost_petrol}, Diesel: NGN{fuel_cost_diesel}")
        logger.info(f"Times - Free: {free_flow_time}min, Congested: {congested_travel_time}min")
        logger.info(f"Distance: {distance_km}km, Analysis Period: {analysis_period}hr")
        logger.info(f"Flow Rate: {flow_rate} veh/hr, Productivity: NGN{productivity_value}/min")
        logger.info("=== END ACTUAL PARAMETERS ===")

        data = []

        for city in cities:
            has_counts = False
            normalized_city = city.lower().replace(' ', '_')
            for v in vehicle_classes:
                vehicle_class = v['name']
                key = v['key']

                count_field = f'Real_Vehicle_Count_{normalized_city}_{key}'
                vor_field = f'Real_VOR_{normalized_city}_{key}'

                count_str = request.form.get(count_field, '')
                vor_str = request.form.get(vor_field, '')

                count = 0
                if count_str:
                    try:
                        count = int(count_str)
                        if count < 0:
                            flash(f"Vehicle count for {vehicle_class} in {city} must be non-negative", "error")
                            return redirect(url_for('data_entry'))
                        if count > 0:
                            has_counts = True
                    except ValueError:
                        flash(f"Vehicle count for {vehicle_class} in {city} must be an integer", "error")
                        return redirect(url_for('data_entry'))

                # Always use the user-provided occupancy rate, never default to 1.0
                vor = 1.0  # Start with default
                if vor_str and vor_str.strip():
                    try:
                        vor = float(vor_str)
                        if vor <= 0:
                            flash(f"Occupancy rate for {vehicle_class} in {city} must be positive", "error")
                            return redirect(url_for('data_entry'))
                    except ValueError:
                        flash(f"Occupancy rate for {vehicle_class} in {city} must be a number", "error")
                        return redirect(url_for('data_entry'))
                else:
                    # If no user input, use the vehicle parameter default (not hardcoded 1.0)
                    vor = model.vehicle_parameters.get(vehicle_class, {}).get('occupancy_avg', 1.0)

                if count > 0 and vor <= 0:
                    flash(f"Occupancy rate for {vehicle_class} in {city} must be positive when count is non-zero",
                          "error")
                    return redirect(url_for('data_entry'))

                emission_factor = (
                    model.vehicle_parameters.get(vehicle_class, {}).get('emission_factor', 0.2)
                    if model.vehicle_parameters.get(vehicle_class, {}).get('fuel_type', 'petrol') == 'petrol'
                    else model.vehicle_parameters.get(vehicle_class, {}).get('emission_factor', 0.7)
                )

                if count > 0:
                    if vehicle_class not in valid_vehicle_types:
                        flash(f"Invalid vehicle type {vehicle_class} for city {city}", "error")
                        return redirect(url_for('data_entry'))

                    # === UPDATED DATA.APPEND() SECTION - Include ALL user inputs ===
                    data.append({
                        "Date": entry_date,
                        "Time": entry_time,
                        "State": state,
                        "City": city,
                        "Road": road_name,
                        "Vehicle_Type": vehicle_class,
                        "Real_Vehicle_Count": count,
                        "Real_VOR": vor,

                        # === CORE USER INPUTS (these must match form exactly) ===
                        "Congested_Travel_Time_Minutes": congested_travel_time,
                        "Free_Flow_Time_Minutes": free_flow_time,
                        "Distance_KM": distance_km,
                        "Value_Per_Minute_Naira": productivity_value,
                        "Fuel_Cost_Per_Liter_Petrol": fuel_cost_petrol,
                        "Fuel_Cost_Per_Liter_Diesel": fuel_cost_diesel,
                        "Emission_Factor_Petrol": emission_factor_petrol,
                        "Emission_Factor_Diesel": emission_factor_diesel,

                        # === ANALYSIS PARAMETERS ===
                        "analysis_period_hr": analysis_period,
                        "flow_rate_veh_per_hr": flow_rate,

                        # === CALCULATED FIELDS ===
                        "Real_Delay_Time": congested_travel_time - free_flow_time,

                        # === ADVANCED FIELDS (user provided or sensible defaults) ===
                        "Free_Flow_Speed_KPH": free_flow_speed,
                        "Congested_Speed_KPH": congested_speed,
                        "Avg_Acceleration": avg_acceleration,
                        "Avg_Deceleration": avg_deceleration,
                        "Idle_Time_Percentage": idle_time_percentage,
                        "Stops_Per_KM": stops_per_km,
                        "Road_Grade": road_grade,
                        "Temperature_C": temperature_c,

                        # === EMISSION MODEL PARAMETERS ===
                        "Emission_Model": emission_model.value,
                        "Barth_Alpha": barth_alpha,
                        "Barth_Beta": barth_beta,
                        "Barth_Gamma": barth_gamma,
                        "Barth_Traffic_Flow": barth_traffic_flow,
                        "Barth_Road_Gradient": barth_road_gradient,
                        "Barth_Acceleration": barth_acceleration,
                        "Vehicle_Emission_Factor": emission_factor,

                        # === VEHICLE DISTRIBUTION ===
                        "class_distribution": class_distributions.get(vehicle_class, 0.0),
                    })

            if not has_counts:
                flash(f"At least one vehicle count is required for {city}.", "error")
                return redirect(url_for('data_entry'))

        if not data:
            flash("Please provide data for at least one city with non-zero vehicle counts", "error")
            return redirect(url_for('data_entry'))

        if all(row['Real_Vehicle_Count'] == 0 for row in data):
            flash("No valid vehicle counts provided", "error")
            return redirect(url_for('data_entry'))

        logger.debug(f"Data for add_city_data: {data}")

        # Create form_data structure instead of saving to CSV
        form_data = {}
        for row in data:
            state = row['State']
            city = row['City']

            if state not in form_data:
                form_data[state] = {}
            if city not in form_data[state]:
                form_data[state][city] = []

            form_data[state][city].append(row)

        # Get the metric from form and convert to enum
        metric_id = request.form.get('metric', 'productivity_loss')
        metric_mapping = {
            'productivity_loss': CalculationMetric.PRODUCTIVITY_LOSS,
            'excess_fuel': CalculationMetric.EXCESS_FUEL,
            'co2_emissions': CalculationMetric.CO2_EMISSIONS
        }
        metric_enum = metric_mapping.get(metric_id, CalculationMetric.PRODUCTIVITY_LOSS)

        metric_display = {
            CalculationMetric.PRODUCTIVITY_LOSS: 'Productivity Loss',
            CalculationMetric.EXCESS_FUEL: 'Excess Fuel Used',
            CalculationMetric.CO2_EMISSIONS: 'CO₂ Emissions'
        }.get(metric_enum, 'Productivity Loss')

        # Validate model selection for the metric
        model_validation_result = model.validate_model_selection(metric_enum, emission_model)
        if not model_validation_result:
            logger.warning(f"Model {emission_model.value} may not be optimal for {metric_enum.value}")

        # Pass form_data directly to analysis
        results, report_data, vehicle_distributions, chart_images, model_used, model = get_analysis_results(
            metric_enum, emission_model, form_data
        )

        # CRITICAL: In your dashboard() POST route, ensure this section exists:
        if not results or not report_data:
            logger.error("Incomplete analysis results")
            flash("Analysis failed to produce complete results", "error")
            return redirect(url_for('data_entry'))

        # Ensure all required variables are set for the template
        if 'report_data' not in locals():
            report_data = []
        if 'vehicle_distributions' not in locals():
            vehicle_distributions = {}
        if 'summary' not in locals():
            summary = {}
        if 'state_results' not in locals():
            state_results = {}
        if 'city_results' not in locals():
            city_results = {}

        # CRITICAL: Store ALL data needed for PDF generation including RAW user inputs
        session['analysis_data'] = {
            'form_data': form_data,
            'metric_enum': metric_enum.value,
            'emission_model': emission_model.value,
            'metric_id': metric_id,
            'results': convert_numpy_types(results),
            'report_data': convert_numpy_types(report_data),
            'vehicle_distributions': convert_numpy_types(vehicle_distributions),
            'chart_images': chart_images,
            # USE THE NEW RAW DATA STRUCTURE:
            'user_inputs': raw_user_inputs  # Store raw user inputs for accurate PDF generation
        }
        # Also store separately for easy access
        session['form_data'] = form_data
        session['metric'] = metric_id
        session['emission_model'] = emission_model.value
        session['current_analysis'] = {
            'metric_id': metric_id,
            'emission_model': emission_model.value,
            'cities': cities,
            'state': state
        }
        session.modified = True

        logger.info(f"✓ Stored analysis data in session for PDF generation: {len(form_data)} states")
        logger.info(
            f"✓ Stored RAW user inputs: analysis_period={analysis_period}, flow_rate={flow_rate}, currency={currency}")
        logger.info(f"✓ Stored RAW vehicle counts: {raw_user_inputs['vehicle_counts']}")

        state_results = results['state_results']
        city_results = results['city_results']

        for state in city_results:
            for city in city_results[state]:
                distribution_key = f"{state}_{city}"
                if distribution_key in vehicle_distributions:
                    city_results[state][city]['vehicle_breakdown'] = vehicle_distributions[distribution_key]
                else:
                    city_results[state][city]['vehicle_breakdown'] = {
                        'Motorcycles': 0,
                        'Cars/SUVs': 0,
                        'Coasters/Buses': 0,
                        'Trucks': 0,
                        'Tankers/Trailers': 0
                    }

        summary = results['total_summary']

        for state in city_results:
            for city, data in city_results[state].items():
                if 'productivity_loss' in data:
                    data['productivity_loss_formatted'] = format_metric_value(
                        data['productivity_loss'], metric_enum
                    )
                data['total_vehicles_formatted'] = f"{data.get('total_vehicles', 0):,.0f} vehicles"
                data['total_people_formatted'] = f"{data.get('total_people', 0):,.0f} people"

        summary['total_vehicles_all_cities_formatted'] = f"{summary.get('total_vehicles_all_cities', 0):,.0f} vehicles"
        summary['total_people_all_cities_formatted'] = f"{summary.get('total_people_all_cities', 0):,.0f} commuters"
        summary['total_metric_formatted'] = format_metric_value(
            summary.get(f'total_{metric_enum.value}_all_cities', 0), metric_enum
        )

        for row in report_data:
            if 'Productivity_Loss' in row:
                row['Productivity_Loss_Formatted'] = format_metric_value(row['Productivity_Loss'], metric_enum)
            if 'Vehicle_Count' in row:
                row['Vehicle_Count_Formatted'] = f"{row['Vehicle_Count']:,.0f} vehicles"

        logger.info("=== VEHICLE BREAKDOWN DATA FOR DASHBOARD CHARTS ===")
        for state in city_results:
            for city, data in city_results[state].items():
                breakdown = data.get('vehicle_breakdown', {})
                logger.info(f"{city}, {state}: vehicle_breakdown = {breakdown}")
                for vehicle_type in ['Motorcycles', 'Cars/SUVs', 'Coasters/Buses', 'Trucks', 'Tankers/Trailers']:
                    if vehicle_type not in breakdown:
                        breakdown[vehicle_type] = 0
                        logger.warning(f"Added missing vehicle type {vehicle_type} for {city}, {state}")
        logger.info("=== END VEHICLE BREAKDOWN DATA ===")

        emission_models_list = model.get_available_models()

        highest_city_name = ""
        max_vehicles = 0
        for state in city_results:
            for city, city_data in city_results[state].items():
                if city_data['total_vehicles'] > max_vehicles:
                    max_vehicles = city_data['total_vehicles']
                    highest_city_name = f"{city}, {state}"

        homepage_data = {
            'title': "Nigeria Traffic Analysis System",
            'subtitle': f"{metric_display} Analysis Report",
            'stats': [
                {'label': 'Total Vehicles Analyzed', 'value': summary['total_vehicles_all_cities_formatted']},
                {'label': 'Total People Affected', 'value': summary['total_people_all_cities_formatted']},
                {'label': f'Total {metric_display}', 'value': summary['total_metric_formatted']}
            ],
            'features': [
                "Accurate vehicle counting and classification",
                "Real-time congestion analysis",
                f"Emission modeling ({model_used.title()})",
                f"{metric_display} assessment"
            ],
            'how_it_works': [
                "Collect traffic data via sensors or manual input",
                f"Analyze vehicle counts and congestion for {metric_display.lower()}",
                f"Calculate {metric_display.lower()} using {model_used.title()} formula",
                "Generate comprehensive reports and visualizations"
            ]
        }

        metrics = [
            {'id': 'productivity_loss', 'name': 'Productivity Loss'},
            {'id': 'excess_fuel', 'name': 'Excess Fuel Used'},
            {'id': 'co2_emissions', 'name': 'CO₂ Emissions'}
        ]
        current_emission_model = model_used

        logger.info(f"Dashboard - Final city results structure: {city_results}")
        for state, cities in city_results.items():
            for city, data in cities.items():
                logger.info(
                    f"Dashboard - Final: {city} in {state}: vehicles={data.get('total_vehicles')}, vehicle_breakdown={data.get('vehicle_breakdown')}")

        return render_template(
            'traffic_dashboard_pdf.html',
            homepage=homepage_data,
            summary=summary,
            state_results=state_results,
            city_results=city_results,
            report_data=report_data,
            vehicle_distributions=vehicle_distributions,
            chart_images={k: f"/temp_charts/{Path(v).name}" for k, v in chart_images.items()},
            emission_models=emission_models_list,
            current_emission_model=current_emission_model,
            current_date=datetime.now().strftime('%Y-%m-%d'),
            current_time=datetime.now().strftime('%H:%M'),
            current_year=datetime.now().year,
            results=results,
            highest_city_name=highest_city_name,
            metric=metric_display,
            metric_id=metric_id,
            metrics=metrics,
            format_metric_value=format_metric_value,
            metric_enum=metric_enum
        )

    except Exception as e:
        logger.error(f"Error processing form data: {str(e)}", exc_info=True)
        flash(f"Error processing input: {str(e)}", "error")
        return redirect(url_for('data_entry'))


@app.route('/generate_report')
def generate_report():
    """Generate PDF report using WeasyPrint HTML-to-PDF."""
    try:
        # Get analysis data from session
        analysis_data = session.get('analysis_data', {})
        if not analysis_data:
            flash("No analysis data found. Please run an analysis first.", "error")
            return redirect(url_for('dashboard_get'))

        # Extract data from session
        results = analysis_data.get('results', {})
        metric_id = analysis_data.get('metric_id', 'co2_emissions')
        emission_model_str = analysis_data.get('emission_model', 'moves')
        form_data = analysis_data.get('form_data', {})
        chart_images = analysis_data.get('chart_images', {})

        # Get the metric enum from session
        metric_enum_value = analysis_data.get('metric_enum')
        if metric_enum_value:
            metric_enum = CalculationMetric(metric_enum_value)
        else:
            metric_enum = CalculationMetric.CO2_EMISSIONS  # Default

        metric_str = {
            CalculationMetric.PRODUCTIVITY_LOSS: 'productivity_loss',
            CalculationMetric.EXCESS_FUEL: 'excess_fuel',
            CalculationMetric.CO2_EMISSIONS: 'co2_emissions'
        }[metric_enum]

        total_metric_key = f'total_{metric_str}_all_cities'
        total_metric = results.get('total_summary', {}).get(total_metric_key, 0)

        total_vehicles = results.get('total_summary', {}).get('total_vehicles_all_cities', 0)
        total_people = results.get('total_summary', {}).get('total_people_all_cities', 0)

        # Get additional parameters from form data or use defaults
        first_city_data = next(iter(form_data.values())) if form_data else {}
        first_vehicle_data = next(iter(first_city_data.values())) if first_city_data else [{}]

        if first_vehicle_data:
            first_row = first_vehicle_data[0]
            fuel_cost_petrol = first_row.get('Fuel_Cost_Per_Liter_Petrol', 900.0)
            fuel_cost_diesel = first_row.get('Fuel_Cost_Per_Liter_Diesel', 1220.0)
            emission_factor_petrol = first_row.get('Emission_Factor_Petrol', 2.31)
            emission_factor_diesel = first_row.get('Emission_Factor_Diesel', 2.68)
            value_per_minute = first_row.get('Value_Per_Minute_Naira', 15.67)
            distance_km = first_row.get('Distance_KM', 8.0)
            free_flow_time = first_row.get('Free_Flow_Time_Minutes', 10.0)
            congested_travel_time = first_row.get('Congested_Travel_Time_Minutes', 35.0)
        else:
            # Default values if no data available
            fuel_cost_petrol = 900.0
            fuel_cost_diesel = 1220.0
            emission_factor_petrol = 2.31
            emission_factor_diesel = 2.68
            value_per_minute = 15.67
            distance_km = 8.0
            free_flow_time = 10.0
            congested_travel_time = 35.0

        # Prepare analysis_data for PDF
        pdf_analysis_data = {
            'total_vehicles': total_vehicles,
            'total_people': total_people,
            f'total_{metric_str}': total_metric,
            'emission_model': emission_model_str.upper(),
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'generation_time': datetime.now().strftime('%H:%M'),
            'fuel_cost_petrol': fuel_cost_petrol,
            'fuel_cost_diesel': fuel_cost_diesel,
            'emission_factor_petrol': emission_factor_petrol,
            'emission_factor_diesel': emission_factor_diesel,
            'value_per_minute': value_per_minute,
            'distance_km': distance_km,
            'free_flow_time': free_flow_time,
            'congested_travel_time': congested_travel_time,
            'chart_images': chart_images
        }

        # Prepare cities_data for PDF
        pdf_cities_data = []
        city_results = results.get('city_results', {})

        for state, cities in city_results.items():
            for city, data in cities.items():
                city_info = {
                    'state': state,
                    'city': city,
                    'total_vehicles': data.get('total_vehicles', 0),
                    'total_people': data.get('total_people', 0),
                    f'total_{metric_str}': data.get(f'total_{metric_str}', 0),
                    'vehicle_details': {}
                }

                # Add vehicle details if available
                vehicle_breakdown = data.get('vehicle_breakdown', {})
                vehicle_metric_dict_key = f'{metric_str}_by_vehicle_type'
                for vehicle_type, count in vehicle_breakdown.items():
                    if count > 0:
                        vehicle_metric_value = data.get(vehicle_metric_dict_key, {}).get(vehicle_type, 0)
                        city_info['vehicle_details'][vehicle_type] = {
                            'count': count,
                            'occupancy': data.get('occupancy_rates', {}).get(vehicle_type, 1.0),
                            metric_str: vehicle_metric_value
                        }

                pdf_cities_data.append(city_info)

        # Generate PDF with WeasyPrint
        pdf_filename = generate_traffic_report_pdf(pdf_analysis_data, pdf_cities_data, metric_enum=metric_enum)

        # Return the PDF
        return send_file(
            pdf_filename,
            as_attachment=True,
            download_name=f"Nigeria_Traffic_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        logger.error(f'Error generating report: {str(e)}', exc_info=True)
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('dashboard_get'))


@app.route('/traffic_report')
def traffic_report():
    """Display the traffic analysis report."""
    # Get analysis data from session
    analysis_data = session.get('analysis_data', {})
    if not analysis_data:
        flash("No analysis data found. Please run an analysis first.", "error")
        return redirect(url_for('dashboard_get'))

    results = analysis_data.get('results', {})
    metric_id = analysis_data.get('metric_id', 'co2_emissions')
    emission_model = analysis_data.get('emission_model', 'moves')

    return render_template(
        'traffic_report.html',
        results=results,
        metric_id=metric_id,
        emission_model=emission_model,
        current_year=datetime.now().year
    )


@app.route('/analysis')
def analysis():
    """Render the analysis page with results and charts."""
    # Get analysis data from session
    analysis_data = session.get('analysis_data', {})
    if not analysis_data:
        flash("No analysis data found. Please run an analysis first.", "error")
        return redirect(url_for('dashboard_get'))

    results = analysis_data.get('results', {})
    metric_id = analysis_data.get('metric_id', 'co2_emissions')
    emission_model = analysis_data.get('emission_model', 'moves')
    chart_images = analysis_data.get('chart_images', {})

    return render_template(
        'analysis.html',
        results=results,
        metric_id=metric_id,
        emission_model=emission_model,
        chart_images=chart_images,
        current_year=datetime.now().year
    )


@app.route('/temp_charts/<filename>')
def serve_temp_chart(filename):
    """Serve temporary chart images."""
    try:
        chart_path = os.path.join('temp_charts', filename)
        if os.path.exists(chart_path):
            return send_file(chart_path, mimetype='image/png')
        else:
            # Return a default image or 404
            return "Chart not found", 404
    except Exception as e:
        logger.error(f"Error serving chart {filename}: {str(e)}")
        return "Error serving chart", 500


@app.route('/analysis_data')
def analysis_data():
    """Provide JSON data for dashboard."""
    analysis_data = session.get('analysis_data', {})
    if not analysis_data:
        return jsonify({'error': 'No analysis data available'})

    return jsonify(analysis_data.get('results', {}))


@app.route('/dashboard', methods=['GET'])
def dashboard_get():
    """Render the dashboard page with proper error handling."""
    # Get analysis data from session
    analysis_data = session.get('analysis_data', {})
    if not analysis_data:
        flash("No analysis data found. Please run an analysis first from the Data Entry page.", "info")
        return redirect(url_for('data_entry'))

    results = analysis_data.get('results', {})
    metric_id = analysis_data.get('metric_id', 'co2_emissions')
    emission_model = analysis_data.get('emission_model', 'moves')
    chart_images = analysis_data.get('chart_images', {})
    report_data = analysis_data.get('report_data', [])  # ADD THIS LINE
    vehicle_distributions = analysis_data.get('vehicle_distributions', {})  # ADD THIS LINE

    # Ensure all required data is present
    if not results or 'city_results' not in results:
        flash("Invalid analysis data. Please run a new analysis.", "error")
        return redirect(url_for('data_entry'))

    # Get metric enum and display name
    metric_enum_value = analysis_data.get('metric_enum')
    if metric_enum_value:
        metric_enum = CalculationMetric(metric_enum_value)
    else:
        metric_enum = CalculationMetric.CO2_EMISSIONS

    metric_display = {
        CalculationMetric.PRODUCTIVITY_LOSS: 'Productivity Loss',
        CalculationMetric.EXCESS_FUEL: 'Excess Fuel Used',
        CalculationMetric.CO2_EMISSIONS: 'CO₂ Emissions'
    }.get(metric_enum, 'Productivity Loss')

    # Prepare summary data
    summary = results.get('total_summary', {})
    state_results = results.get('state_results', {})
    city_results = results.get('city_results', {})

    # Format summary data
    summary['total_vehicles_all_cities_formatted'] = f"{summary.get('total_vehicles_all_cities', 0):,.0f} vehicles"
    summary['total_people_all_cities_formatted'] = f"{summary.get('total_people_all_cities', 0):,.0f} commuters"

    # Get the correct metric key based on the current metric
    metric_str = {
        CalculationMetric.PRODUCTIVITY_LOSS: 'productivity_loss',
        CalculationMetric.EXCESS_FUEL: 'excess_fuel',
        CalculationMetric.CO2_EMISSIONS: 'co2_emissions'
    }[metric_enum]

    total_metric_key = f'total_{metric_str}_all_cities'
    summary['total_metric_formatted'] = format_metric_value(
        summary.get(total_metric_key, 0), metric_enum
    )

    # Format city results
    for state in city_results:
        for city, data in city_results[state].items():
            if 'productivity_loss' in data:
                data['productivity_loss_formatted'] = format_metric_value(
                    data['productivity_loss'], metric_enum
                )
            data['total_vehicles_formatted'] = f"{data.get('total_vehicles', 0):,.0f} vehicles"
            data['total_people_formatted'] = f"{data.get('total_people', 0):,.0f} people"

    # Format report data
    for row in report_data:
        if 'Productivity_Loss' in row:
            row['Productivity_Loss_Formatted'] = format_metric_value(row['Productivity_Loss'], metric_enum)
        if 'Vehicle_Count' in row:
            row['Vehicle_Count_Formatted'] = f"{row['Vehicle_Count']:,.0f} vehicles"

    # Prepare homepage data
    homepage_data = {
        'title': "Nigeria Traffic Analysis System",
        'subtitle': f"{metric_display} Analysis Report",
        'stats': [
            {'label': 'Total Vehicles Analyzed', 'value': summary['total_vehicles_all_cities_formatted']},
            {'label': 'Total People Affected', 'value': summary['total_people_all_cities_formatted']},
            {'label': f'Total {metric_display}', 'value': summary['total_metric_formatted']}
        ],
        'features': [
            "Accurate vehicle counting and classification",
            "Real-time congestion analysis",
            f"Emission modeling ({emission_model.title()})",
            f"{metric_display} assessment"
        ],
        'how_it_works': [
            "Collect traffic data via sensors or manual input",
            f"Analyze vehicle counts and congestion for {metric_display.lower()}",
            f"Calculate {metric_display.lower()} using {emission_model.title()} formula",
            "Generate comprehensive reports and visualizations"
        ]
    }

    metrics = [
        {'id': 'productivity_loss', 'name': 'Productivity Loss'},
        {'id': 'excess_fuel', 'name': 'Excess Fuel Used'},
        {'id': 'co2_emissions', 'name': 'CO₂ Emissions'}
    ]

    # Find highest city
    highest_city_name = ""
    max_vehicles = 0
    for state in city_results:
        for city, city_data in city_results[state].items():
            if city_data['total_vehicles'] > max_vehicles:
                max_vehicles = city_data['total_vehicles']
                highest_city_name = f"{city}, {state}"

    return render_template(
        'traffic_dashboard_pdf.html',
        homepage=homepage_data,
        summary=summary,
        state_results=state_results,
        city_results=city_results,
        report_data=report_data,  # NOW DEFINED
        vehicle_distributions=vehicle_distributions,  # NOW DEFINED
        chart_images={k: f"/temp_charts/{Path(v).name}" for k, v in chart_images.items()},
        emission_models=emission_models,
        current_emission_model=emission_model,
        current_date=datetime.now().strftime('%Y-%m-%d'),
        current_time=datetime.now().strftime('%H:%M'),
        current_year=datetime.now().year,
        results=results,
        highest_city_name=highest_city_name,
        metric=metric_display,
        metric_id=metric_id,
        metrics=metrics,
        format_metric_value=format_metric_value,
        metric_enum=metric_enum
    )


@app.route('/download_report')
def download_report():
    """Download the CSV report - kept for template compatibility."""
    try:
        analysis_data = session.get('analysis_data', {})
        if not analysis_data:
            flash("No analysis data found", "error")
            return redirect(url_for('dashboard_get'))

        report_data = analysis_data.get('report_data', [])
        if not report_data:
            flash("No report data available", "error")
            return redirect(url_for('dashboard_get'))

        # Create DataFrame from report data
        df = pd.DataFrame(report_data)

        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            as_attachment=True,
            download_name=f"traffic_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mimetype='text/csv'
        )

    except Exception as e:
        logger.error(f"Error downloading CSV report: {str(e)}")
        flash(f"Error downloading report: {str(e)}", "error")
        return redirect(url_for('dashboard_get'))


@app.route('/download_csv')
def download_csv():
    """Download the CSV report (alias for download_report)."""
    return download_report()


@app.route('/download_traffic_data')
def download_traffic_data():
    """Download the traffic data - kept for template compatibility."""
    try:
        form_data = session.get('form_data', {})
        if not form_data:
            flash("No traffic data found", "error")
            return redirect(url_for('dashboard_get'))

        # Flatten the form data structure
        flat_data = []
        for state, cities in form_data.items():
            for city, vehicles in cities.items():
                for vehicle in vehicles:
                    flat_data.append(vehicle)

        if not flat_data:
            flash("No traffic data available", "error")
            return redirect(url_for('dashboard_get'))

        # Create DataFrame
        df = pd.DataFrame(flat_data)

        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            as_attachment=True,
            download_name=f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mimetype='text/csv'
        )

    except Exception as e:
        logger.error(f"Error downloading traffic data: {str(e)}")
        flash(f"Error downloading traffic data: {str(e)}", "error")
        return redirect(url_for('dashboard_get'))


@app.route('/download_pdf')
def download_pdf():
    """Generate and download the PDF report using WeasyPrint."""
    try:
        # Get analysis data from session
        analysis_data = session.get('analysis_data', {})
        if not analysis_data:
            flash("No analysis data found. Please run an analysis first.", "error")
            return redirect(url_for('dashboard_get'))

        # Extract data from session
        results = analysis_data.get('results', {})
        metric_id = analysis_data.get('metric_id', 'co2_emissions')
        emission_model_str = analysis_data.get('emission_model', 'moves')
        user_inputs = analysis_data.get('user_inputs', {})  # GET USER INPUTS
        chart_images = analysis_data.get('chart_images', {})

        # Get the metric enum from session
        metric_enum_value = analysis_data.get('metric_enum')
        if metric_enum_value:
            metric_enum = CalculationMetric(metric_enum_value)
        else:
            metric_enum = CalculationMetric.CO2_EMISSIONS  # Default

        metric_str = {
            CalculationMetric.PRODUCTIVITY_LOSS: 'productivity_loss',
            CalculationMetric.EXCESS_FUEL: 'excess_fuel',
            CalculationMetric.CO2_EMISSIONS: 'co2_emissions'
        }[metric_enum]

        total_metric_key = f'total_{metric_str}_all_cities'
        total_metric = results.get('total_summary', {}).get(total_metric_key, 0)

        total_vehicles = results.get('total_summary', {}).get('total_vehicles_all_cities', 0)
        total_people = results.get('total_summary', {}).get('total_people_all_cities', 0)

        # Prepare analysis_data for PDF - USE USER INPUTS
        pdf_analysis_data = {
            'total_vehicles': total_vehicles,
            'total_people': total_people,
            f'total_{metric_str}': total_metric,
            'emission_model': emission_model_str.upper(),
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'generation_time': datetime.now().strftime('%H:%M'),
            'chart_images': chart_images,
            'user_inputs': user_inputs  # PASS USER INPUTS TO PDF
        }

        # Prepare cities_data for PDF
        pdf_cities_data = []
        city_results = results.get('city_results', {})

        for state, cities in city_results.items():
            for city, data in cities.items():
                city_info = {
                    'state': state,
                    'city': city,
                    'total_vehicles': data.get('total_vehicles', 0),
                    'total_people': data.get('total_people', 0),
                    f'total_{metric_str}': data.get(f'total_{metric_str}', 0),
                    'vehicle_details': {}
                }

                # Add vehicle details if available
                vehicle_breakdown = data.get('vehicle_breakdown', {})
                vehicle_metric_dict_key = f'{metric_str}_by_vehicle_type'
                for vehicle_type, count in vehicle_breakdown.items():
                    if count > 0:
                        vehicle_metric_value = data.get(vehicle_metric_dict_key, {}).get(vehicle_type, 0)
                        city_info['vehicle_details'][vehicle_type] = {
                            'count': count,
                            'occupancy': data.get('occupancy_rates', {}).get(vehicle_type, 1.0),
                            metric_str: vehicle_metric_value
                        }

                pdf_cities_data.append(city_info)

        # Generate PDF with WeasyPrint
        pdf_filename = generate_traffic_report_pdf(pdf_analysis_data, pdf_cities_data, metric_enum=metric_enum)

        # Return the PDF
        return send_file(
            pdf_filename,
            as_attachment=True,
            download_name=f"Nigeria_Traffic_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        logger.error(f'Error generating PDF: {str(e)}', exc_info=True)
        flash(f'Error generating PDF: {str(e)}', 'error')
        return redirect(url_for('dashboard_get'))


@app.route('/clear_data', methods=['POST'])
def clear_data():
    """Clear all temporary data."""
    try:
        # Clear session data
        session.pop('analysis_data', None)
        session.pop('form_data', None)
        session.pop('metric', None)
        session.pop('emission_model', None)
        session.pop('current_analysis', None)

        # Clear temporary chart files
        temp_charts_dir = 'temp_charts'
        if os.path.exists(temp_charts_dir):
            for file in os.listdir(temp_charts_dir):
                if file.endswith('.png'):
                    os.remove(os.path.join(temp_charts_dir, file))

        flash("All data cleared successfully", "success")
        return redirect(url_for('welcome'))

    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        flash(f"Error clearing data: {str(e)}", "error")
        return redirect(url_for('dashboard_get'))


@app.route('/debug_context', methods=['GET'])
def debug_context():
    """Return template context for debugging."""
    analysis_data = session.get('analysis_data', {})
    return jsonify({
        'session_keys': list(session.keys()),
        'analysis_data_keys': list(analysis_data.keys()) if analysis_data else [],
        'results_keys': list(analysis_data.get('results', {}).keys()) if analysis_data.get('results') else []
    })


@app.route('/debug_charts')
def debug_charts():
    """Debug endpoint to check chart status."""
    temp_charts_dir = 'temp_charts'
    if os.path.exists(temp_charts_dir):
        charts = os.listdir(temp_charts_dir)
        return jsonify({'charts': charts, 'count': len(charts)})
    else:
        return jsonify({'charts': [], 'count': 0, 'error': 'temp_charts directory not found'})


# Production configuration
if __name__ == "__main__":
    # Check if we're in production (Render sets PORT environment variable)
    port = int(os.environ.get("PORT", 5000))

    # In production, use 0.0.0.0 to bind to all available interfaces
    host = "0.0.0.0" if os.environ.get("PORT") else "localhost"

    print(f"Starting Nigeria Traffic Analysis System...")
    print(f"Visit http://{host}:{port} to access the application")

    app.run(host=host, port=port, debug=False)