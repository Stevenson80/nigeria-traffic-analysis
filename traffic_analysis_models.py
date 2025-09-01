#!/usr/bin/env python3
"""
TrafficAnalysisModel class for analyzing traffic data and generating rich PDF reports using ReportLab and Matplotlib.
"""
import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
from enum import Enum
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_analysis_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Vehicle Classes Constant (aligned with app.py: Class 1 to Class 10)
VEHICLE_CLASSES = {
    f"Class {i}": {
        "name": f"Vehicle Class {i}",
        "occupancy": 1.5 + i * 2.0,  # Example: increasing occupancy from 1.5 to 20.5
        "fuel_type": "petrol" if i % 2 == 1 else "diesel"  # Alternating fuel types
    } for i in range(1, 11)
}

class EmissionModelType(Enum):
    BASIC = "basic"
    BARTH = "barth"
    MOVES = "moves"

class TrafficAnalysisModel:
    # Constants (aligned with app.py for 2026)
    FUEL_PRICE_PETROL = 1045.0  # Naira/liter
    FUEL_PRICE_DIESEL = 1210.0  # Naira/liter
    EMISSION_FACTOR_PETROL = 2.31  # kg CO2/liter
    EMISSION_FACTOR_DIESEL = 2.68
    VALUE_PER_MINUTE = 8.8  # Naira/min
    CORRIDOR_LENGTH = 6.0  # km
    BARTH_IDLE_FUEL_CONSUMPTION_PETROL = 0.6  # liters/hour
    BARTH_IDLE_FUEL_CONSUMPTION_DIESEL = 0.8
    BARTH_ACCEL_FACTOR_PETROL = 0.00035
    BARTH_ACCEL_FACTOR_DIESEL = 0.00045
    MOVES_BASE_EMISSION_RATE_PETROL = 2.0  # g/km
    MOVES_BASE_EMISSION_RATE_DIESEL = 2.5
    MOVES_SPEED_CORRECTION_FACTOR = 0.05
    MOVES_ACCEL_CORRECTION_FACTOR = 0.1

    BARTH_COEFFICIENTS = {
        'alpha': 0.0003,
        'beta': 0.0025,
        'gamma': 0.0018,
        'traffic_flow': 0.15,
        'road_gradient': 0.10,
        'acceleration': 0.04
    }

    VEHICLE_EMISSION_FACTORS = {f"Class {i}": 0.2 + i * 0.05 for i in range(1, 11)}  # Example factors

    def __init__(self, csv_file_path: Optional[str] = None,
                 emission_model: EmissionModelType = EmissionModelType.BASIC):
        """Initialize the TrafficAnalysisModel with optional CSV data and emission model."""
        self.vehicle_parameters = {
            f"Class {i}": {
                'name': VEHICLE_CLASSES[f"Class {i}"]['name'],
                'occupancy_avg': VEHICLE_CLASSES[f"Class {i}"]['occupancy'],
                'fuel_type': VEHICLE_CLASSES[f"Class {i}"]['fuel_type'],
                'fuel_consumption_l_per_km_free_flow': 0.03 + i * 0.02,
                'fuel_consumption_l_per_km_congested': 0.045 + i * 0.03,
                'weight_kg': 150 + i * 1000,
                'engine_displacement_cc': 125 + i * 500,
                'euro_standard': 3 if i <= 5 else 2,
                'emission_factor': self.VEHICLE_EMISSION_FACTORS[f"Class {i}"]
            } for i in range(1, 11)
        }
        self.value_per_minute = self.VALUE_PER_MINUTE
        self.data = None
        self.road_data = {}
        self.analysis_date = datetime.now().strftime('%Y-%m-%d')
        self.analysis_time = datetime.now().strftime('%H:%M')
        self.emission_model = emission_model
        if csv_file_path:
            self.load_csv_data(csv_file_path)
        else:
            self.load_hardcoded_data()

    def load_hardcoded_data(self):
        """Load hardcoded traffic data for testing when no CSV is provided."""
        try:
            self.road_data = {
                'Nyanya Road': {f"Class {i}": 100 + i * 10 for i in range(1, 11)},
                'Lugbe Road': {f"Class {i}": 80 + i * 12 for i in range(1, 11)},
                'Kubwa Road': {f"Class {i}": 120 + i * 8 for i in range(1, 11)}
            }
            self.data = pd.DataFrame([
                {
                    'Date': self.analysis_date,
                    'Time': self.analysis_time,
                    'Road': road,
                    'Vehicle Type': vehicle_class,
                    'Real_Vehicle_Count': count,
                    'Real_VOR': self.vehicle_parameters[vehicle_class]['occupancy_avg'],
                    'Congested_Travel_Time_Minutes': 45.0,
                    'Distance_KM': self.CORRIDOR_LENGTH,
                    'Free_Flow_Time_Minutes': 4.0,
                    'Value_Per_Minute_Naira': self.VALUE_PER_MINUTE,
                    'Fuel_Cost_Per_Liter_Petrol': self.FUEL_PRICE_PETROL,
                    'Fuel_Cost_Per_Liter_Diesel': self.FUEL_PRICE_DIESEL,
                    'Emission_Factor_Petrol': self.EMISSION_FACTOR_PETROL,
                    'Emission_Factor_Diesel': self.EMISSION_FACTOR_DIESEL,
                    'Free_Flow_Speed_KPH': 60.0,
                    'Congested_Speed_KPH': 8.0,
                    'Avg_Acceleration': 0.5,
                    'Avg_Deceleration': 0.5,
                    'Idle_Time_Percentage': 0.3,
                    'Stops_Per_KM': 2.0,
                    'Road_Grade': 0.0,
                    'Temperature_C': 25.0,
                    'Emission_Model': self.emission_model.value,
                    'Barth_Alpha': self.BARTH_COEFFICIENTS['alpha'],
                    'Barth_Beta': self.BARTH_COEFFICIENTS['beta'],
                    'Barth_Gamma': self.BARTH_COEFFICIENTS['gamma'],
                    'Barth_Traffic_Flow': self.BARTH_COEFFICIENTS['traffic_flow'],
                    'Barth_Road_Gradient': self.BARTH_COEFFICIENTS['road_gradient'],
                    'Barth_Acceleration': self.BARTH_COEFFICIENTS['acceleration'],
                    'Vehicle_Emission_Factor': self.vehicle_parameters[vehicle_class]['emission_factor']
                }
                for road in self.road_data
                for vehicle_class, count in self.road_data[road].items()
            ])
            logger.info("Loaded hardcoded data successfully")
        except Exception as e:
            logger.error(f"Error loading hardcoded data: {str(e)}", exc_info=True)
            self.road_data = {}
            self.data = pd.DataFrame()

    def load_csv_data(self, csv_file_path: str):
        """Load and validate traffic data from CSV file."""
        try:
            csv_path = Path(csv_file_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file {csv_file_path} not found")
            self.data = pd.read_csv(csv_file_path)
            required_columns = ['Road', 'Vehicle Type', 'Real_Vehicle_Count']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"CSV missing required columns: {missing_columns}")

            if not pd.api.types.is_numeric_dtype(self.data['Real_Vehicle_Count']):
                raise ValueError("'Real_Vehicle_Count' must be numeric")
            if (self.data['Real_Vehicle_Count'] < 0).any():
                raise ValueError("'Real_Vehicle_Count' contains negative values")

            numeric_columns = [
                'Congested_Travel_Time_Minutes', 'Distance_KM', 'Free_Flow_Time_Minutes',
                'Free_Flow_Speed_KPH', 'Congested_Speed_KPH', 'Avg_Acceleration',
                'Avg_Deceleration', 'Idle_Time_Percentage', 'Stops_Per_KM',
                'Road_Grade', 'Temperature_C'
            ]

            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0.0).astype(float)
                else:
                    self.data[col] = 0.0

            default_columns = {
                'Congested_Travel_Time_Minutes': 45.0,
                'Distance_KM': self.CORRIDOR_LENGTH,
                'Free_Flow_Time_Minutes': 4.0,
                'Free_Flow_Speed_KPH': 60.0,
                'Congested_Speed_KPH': 8.0,
                'Avg_Acceleration': 0.5,
                'Avg_Deceleration': 0.5,
                'Idle_Time_Percentage': 0.3,
                'Stops_Per_KM': 2.0,
                'Road_Grade': 0.0,
                'Temperature_C': 25.0
            }

            for col, default in default_columns.items():
                if col not in self.data.columns:
                    self.data[col] = default

            for col, default in self.BARTH_COEFFICIENTS.items():
                col_name = f"Barth_{col.capitalize()}"
                if col_name not in self.data.columns:
                    self.data[col_name] = default

            if 'Vehicle_Emission_Factor' not in self.data.columns:
                self.data['Vehicle_Emission_Factor'] = self.data['Vehicle Type'].map(
                    lambda x: self.vehicle_parameters.get(x, {}).get('emission_factor', 0.2))

            self.road_data = {}
            for road in self.data['Road'].unique():
                road_df = self.data[self.data['Road'] == road]
                self.road_data[road] = {
                    row['Vehicle Type']: row['Real_Vehicle_Count']
                    for _, row in road_df.iterrows()
                }
            logger.info(f"Loaded and validated data from {csv_file_path}")
        except Exception as e:
            logger.error(f"Error loading CSV data from {csv_file_path}: {str(e)}", exc_info=True)
            self.data = pd.DataFrame()
            self.road_data = {}

    def get_vehicle_distribution(self, road_name: str) -> List[Dict[str, Any]]:
        """Get vehicle distribution for a specific road."""
        try:
            if road_name not in self.road_data:
                logger.warning(f"No data for road: {road_name}")
                return []
            return [
                {'vehicle_type': VEHICLE_CLASSES[vehicle]['name'], 'count': int(count) if not pd.isna(count) else 0}
                for vehicle, count in self.road_data[road_name].items() if count > 0
            ]
        except Exception as e:
            logger.error(f"Error getting vehicle distribution for {road_name}: {str(e)}", exc_info=True)
            return []

    def calculate_fuel_consumption_barth(self, avg_speed_kmh: float, vehicle_type: str, distance: float, count: int) -> Tuple[float, float]:
        """Calculate fuel consumption using Barth's comprehensive model."""
        try:
            vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
            if not vehicle_params:
                logger.warning(f"No parameters for vehicle type: {vehicle_type}")
                return 0.0, 0.0

            weight = vehicle_params.get('weight_kg', 1000)
            engine_displacement = vehicle_params.get('engine_displacement_cc', 1500)
            fuel_type = vehicle_params.get('fuel_type', 'petrol')
            accel_factor = self.BARTH_ACCEL_FACTOR_PETROL if fuel_type == 'petrol' else self.BARTH_ACCEL_FACTOR_DIESEL
            idle_consumption = self.BARTH_IDLE_FUEL_CONSUMPTION_PETROL if fuel_type == 'petrol' else self.BARTH_IDLE_FUEL_CONSUMPTION_DIESEL

            alpha = self.BARTH_COEFFICIENTS['alpha']
            beta = self.BARTH_COEFFICIENTS['beta']
            gamma = self.BARTH_COEFFICIENTS['gamma']
            traffic_flow = self.BARTH_COEFFICIENTS['traffic_flow']
            road_gradient = self.BARTH_COEFFICIENTS['road_gradient']
            acceleration = self.BARTH_COEFFICIENTS['acceleration']

            base_consumption = (alpha * engine_displacement + beta * weight + gamma * road_gradient) * distance
            congestion_factor = max(0.1, min(1.0, 60.0 / avg_speed_kmh if avg_speed_kmh > 0 else 1.0))
            traffic_effect = base_consumption * traffic_flow * congestion_factor
            acceleration_events = congestion_factor * 10
            acceleration_effect = accel_factor * acceleration_events * distance
            idle_time_hours = 0.3 * (distance / avg_speed_kmh) if avg_speed_kmh > 0 else 0.3
            idle_fuel = idle_consumption * idle_time_hours
            total_fuel_consumption = (base_consumption + traffic_effect + acceleration_effect + idle_fuel) * count
            free_flow_consumption = (alpha * engine_displacement + beta * weight) * distance * 0.7 * count
            return free_flow_consumption, total_fuel_consumption
        except Exception as e:
            logger.error(f"Error calculating Barth fuel consumption for {vehicle_type}: {str(e)}", exc_info=True)
            return 0.0, 0.0

    def calculate_fuel_consumption_moves(self, avg_speed_kmh: float, stops_per_km: float, road_grade: float,
                                        temperature_c: float, vehicle_type: str, distance: float, count: int) -> Tuple[float, float]:
        """Calculate fuel consumption using MOVES approximation model."""
        try:
            vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
            if not vehicle_params:
                logger.warning(f"No parameters for vehicle type: {vehicle_type}")
                return 0.0, 0.0

            fuel_type = vehicle_params.get('fuel_type', 'petrol')
            base_rate = vehicle_params.get('fuel_consumption_l_per_km_congested', 0.1)
            speed_factor = 1.0 + self.MOVES_SPEED_CORRECTION_FACTOR * (avg_speed_kmh - 30.0) / 30.0
            stop_factor = 1.0 + 0.1 * stops_per_km
            grade_factor = 1.0 + 0.05 * road_grade
            temp_factor = 1.0 + 0.02 * (temperature_c - 25.0) / 25.0
            total_fuel_consumption = base_rate * distance * speed_factor * stop_factor * grade_factor * temp_factor * count
            free_flow_base = vehicle_params.get('fuel_consumption_l_per_km_free_flow', 0.07)
            free_flow_consumption = free_flow_base * distance * count
            return free_flow_consumption, total_fuel_consumption
        except Exception as e:
            logger.error(f"Error calculating MOVES fuel consumption for {vehicle_type}: {str(e)}", exc_info=True)
            return 0.0, 0.0

    def calculate_barth_emissions(self, vehicle_type: str, count: int, distance: float,
                                 speed: float, acceleration: float, idle_time: float) -> float:
        """Calculate emissions using Barth's model."""
        try:
            vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
            if not vehicle_params:
                logger.warning(f"No parameters for vehicle type: {vehicle_type}")
                return 0.0
            alpha = self.BARTH_COEFFICIENTS['alpha']
            beta = self.BARTH_COEFFICIENTS['beta']
            gamma = self.BARTH_COEFFICIENTS['gamma']
            traffic_flow = self.BARTH_COEFFICIENTS['traffic_flow']
            acceleration_factor = self.BARTH_COEFFICIENTS['acceleration']
            road_gradient = self.BARTH_COEFFICIENTS['road_gradient']
            base_emissions = (
                alpha * vehicle_params.get('engine_displacement_cc', 1500) +
                beta * vehicle_params.get('weight_kg', 1000) +
                gamma * road_gradient +
                traffic_flow * speed +
                acceleration_factor * acceleration
            )
            idle_emissions = 0
            if idle_time > 0:
                if vehicle_params.get('fuel_type', 'petrol') == 'petrol':
                    idle_emissions = self.BARTH_IDLE_FUEL_CONSUMPTION_PETROL * idle_time * self.EMISSION_FACTOR_PETROL
                else:
                    idle_emissions = self.BARTH_IDLE_FUEL_CONSUMPTION_DIESEL * idle_time * self.EMISSION_FACTOR_DIESEL
            total_emissions = (base_emissions * distance + idle_emissions) * count
            return total_emissions
        except Exception as e:
            logger.error(f"Error calculating Barth emissions for {vehicle_type}: {str(e)}", exc_info=True)
            return 0.0

    def calculate_moves_emissions(self, vehicle_type: str, count: int, distance: float,
                                 speed: float, acceleration: float) -> float:
        """Calculate emissions using MOVES-like model."""
        try:
            vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
            if not vehicle_params:
                logger.warning(f"No parameters for vehicle type: {vehicle_type}")
                return 0.0
            if vehicle_params.get('fuel_type', 'petrol') == 'petrol':
                base_rate = self.MOVES_BASE_EMISSION_RATE_PETROL
            else:
                base_rate = self.MOVES_BASE_EMISSION_RATE_DIESEL
            speed_correction = self.MOVES_SPEED_CORRECTION_FACTOR * speed
            accel_correction = self.MOVES_ACCEL_CORRECTION_FACTOR * acceleration
            emissions = base_rate * (1 + speed_correction) * (1 + accel_correction) * distance * count
            return emissions
        except Exception as e:
            logger.error(f"Error calculating MOVES emissions for {vehicle_type}: {str(e)}", exc_info=True)
            return 0.0

    def calculate_basic_emissions(self, vehicle_type: str, count: int, distance: float) -> float:
        """Calculate emissions using basic model (fuel consumption * emission factor)."""
        try:
            vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
            if not vehicle_params:
                logger.warning(f"No parameters for vehicle type: {vehicle_type}")
                return 0.0
            fuel_consumption = vehicle_params.get('fuel_consumption_l_per_km_congested', 0.1) * distance
            emission_factor = vehicle_params.get('emission_factor', 0.2)
            emissions = fuel_consumption * emission_factor * count
            return emissions
        except Exception as e:
            logger.error(f"Error calculating basic emissions for {vehicle_type}: {str(e)}", exc_info=True)
            return 0.0

    def analyze_all_roads(self) -> Dict[str, Any]:
        """Analyze traffic data for all roads using class-based data structure."""
        try:
            if self.data.empty or not self.road_data:
                raise ValueError("No valid data available for analysis")
            road_results = {}
            total_summary = {
                'total_vehicles_all_roads': 0,
                'total_people_all_roads': 0,
                'total_delay_hours_all_roads': 0.0,
                'total_excess_fuel_all_roads': 0.0,
                'total_fuel_cost_all_roads': 0.0,
                'total_co2_all_roads': 0.0,
                'total_productivity_loss_all_roads': 0.0
            }
            for road in self.road_data:
                road_data = self.data[self.data['Road'] == road]
                if road_data.empty:
                    logger.warning(f"No data for road: {road}")
                    continue
                total_vehicles = road_data['Real_Vehicle_Count'].sum()
                total_people = sum(
                    row['Real_Vehicle_Count'] * self.vehicle_parameters.get(row['Vehicle Type'], {}).get(
                        'occupancy_avg', 1.0)
                    for _, row in road_data.iterrows()
                )
                total_excess_fuel = 0
                total_fuel_cost = 0
                total_co2 = 0.0
                for _, row in road_data.iterrows():
                    vehicle_class = row['Vehicle Type']
                    count = row['Real_Vehicle_Count']
                    distance = row['Distance_KM']
                    speed = row.get('Congested_Speed_KPH', 8.0)
                    acceleration = row.get('Avg_Acceleration', 0.5)
                    idle_time = row.get('Idle_Time_Percentage', 0.3) * row.get('Congested_Travel_Time_Minutes', 45.0) / 60.0
                    stops_per_km = row.get('Stops_Per_KM', 2.0)
                    road_grade = row.get('Road_Grade', 0.0)
                    temperature_c = row.get('Temperature_C', 25.0)

                    if self.emission_model == EmissionModelType.BARTH:
                        free_flow_fuel, congested_fuel = self.calculate_fuel_consumption_barth(
                            avg_speed_kmh=speed, vehicle_type=vehicle_class, distance=distance, count=count
                        )
                    elif self.emission_model == EmissionModelType.MOVES:
                        free_flow_fuel, congested_fuel = self.calculate_fuel_consumption_moves(
                            avg_speed_kmh=speed, stops_per_km=stops_per_km, road_grade=road_grade,
                            temperature_c=temperature_c, vehicle_type=vehicle_class, distance=distance, count=count
                        )
                    else:  # BASIC model
                        vehicle_params = self.vehicle_parameters.get(vehicle_class, {})
                        free_flow_fuel = vehicle_params.get('fuel_consumption_l_per_km_free_flow', 0.07) * distance * count
                        congested_fuel = vehicle_params.get('fuel_consumption_l_per_km_congested', 0.1) * distance * count

                    excess_fuel = congested_fuel - free_flow_fuel
                    total_excess_fuel += excess_fuel
                    vehicle_params = self.vehicle_parameters.get(vehicle_class, {})
                    if not vehicle_params:
                        continue
                    if vehicle_params.get('fuel_type', 'petrol') == 'petrol':
                        fuel_cost = excess_fuel * self.FUEL_PRICE_PETROL
                    else:
                        fuel_cost = excess_fuel * self.FUEL_PRICE_DIESEL
                    total_fuel_cost += fuel_cost

                    if self.emission_model == EmissionModelType.BARTH:
                        total_co2 += self.calculate_barth_emissions(
                            vehicle_type=vehicle_class, count=count, distance=distance, speed=speed,
                            acceleration=acceleration, idle_time=idle_time
                        )
                    elif self.emission_model == EmissionModelType.MOVES:
                        total_co2 += self.calculate_moves_emissions(
                            vehicle_type=vehicle_class, count=count, distance=distance, speed=speed,
                            acceleration=acceleration
                        )
                    else:
                        total_co2 += self.calculate_basic_emissions(vehicle_type=vehicle_class, count=count, distance=distance)

                total_productivity_loss = total_people * (
                    road_data['Congested_Travel_Time_Minutes'].iloc[0] -
                    road_data['Free_Flow_Time_Minutes'].iloc[0]
                ) * self.VALUE_PER_MINUTE
                road_results[road] = {
                    'total_vehicles': int(total_vehicles),
                    'total_people': int(total_people),
                    'total_excess_fuel_l': round(total_excess_fuel, 2),
                    'total_co2_kg': round(total_co2, 2),
                    'total_fuel_cost_naira': round(total_fuel_cost, 2),
                    'total_productivity_loss_naira': round(total_productivity_loss, 2)
                }
                total_summary['total_vehicles_all_roads'] += total_vehicles
                total_summary['total_people_all_roads'] += total_people
                total_summary['total_excess_fuel_all_roads'] += total_excess_fuel
                total_summary['total_co2_all_roads'] += total_co2
                total_summary['total_fuel_cost_all_roads'] += total_fuel_cost
                total_summary['total_productivity_loss_all_roads'] += total_productivity_loss
                total_summary['total_delay_hours_all_roads'] += (
                    road_data['Congested_Travel_Time_Minutes'].iloc[0] -
                    road_data['Free_Flow_Time_Minutes'].iloc[0]
                ) / 60.0 * total_people
            total_summary = {k: round(v, 2) if isinstance(v, float) else int(v) for k, v in total_summary.items()}
            return {'road_results': road_results, 'total_summary': total_summary}
        except Exception as e:
            logger.error(f"Error analyzing roads: {str(e)}", exc_info=True)
            return {'road_results': {}, 'total_summary': {}}

    def generate_report(self) -> pd.DataFrame:
        """Generate a detailed report DataFrame."""
        try:
            results = self.analyze_all_roads()
            if not results['road_results']:
                logger.warning("No analysis results available for report generation")
                return pd.DataFrame()
            data = []
            for road, road_data in results['road_results'].items():
                data.append({
                    'Road': road,
                    'Vehicle Count': int(road_data['total_vehicles']),
                    'People Affected': int(road_data['total_people']),
                    'Excess Fuel (L)': round(road_data['total_excess_fuel_l'], 2),
                    'CO2 Emissions (kg)': round(road_data['total_co2_kg'], 2),
                    'Fuel Cost (Naira)': round(road_data['total_fuel_cost_naira'], 2),
                    'Productivity Loss (Naira)': round(road_data['total_productivity_loss_naira'], 2),
                    'Total Economic Impact (Naira)': round(
                        road_data['total_fuel_cost_naira'] + road_data['total_productivity_loss_naira'], 2),
                    'Emission Model': self.emission_model.value
                })
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def get_available_models(self) -> List[str]:
        """Return list of available emission models."""
        return [model.value for model in EmissionModelType]

    def generate_chart_images(self) -> Dict[str, str]:
        """Generate chart images as temporary files for PDF report."""
        try:
            chart_images = {}
            if not self.road_data:
                logger.warning("No road data available for chart generation")
                return chart_images

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            temp_dir = Path('temp_charts')
            temp_dir.mkdir(exist_ok=True)

            for road_name in self.road_data.keys():
                vehicle_dist = self.get_vehicle_distribution(road_name)
                if not vehicle_dist:
                    logger.warning(f"No vehicle distribution data for {road_name}")
                    continue
                vehicle_types = [item["vehicle_type"] for item in vehicle_dist]
                counts = [item["count"] for item in vehicle_dist]
                plt.figure(figsize=(8, 6))
                plt.pie(counts, labels=vehicle_types, autopct='%1.1f%%', colors=colors[:len(vehicle_types)],
                        startangle=90, textprops={'fontsize': 12})
                plt.title(f"{road_name} Vehicle Distribution", fontsize=14, pad=15, weight='bold')
                chart_path = temp_dir / f"pie_{road_name.replace(' ', '_').lower()}.png"
                plt.savefig(chart_path, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                chart_images[f"pie_{road_name.replace(' ', '_').lower()}"] = str(chart_path)
                logger.info(f"Generated pie chart for {road_name} at {chart_path}")

            results = self.analyze_all_roads()
            roads = list(self.road_data.keys())

            people_data = [results['road_results'][road]['total_people'] for road in roads]
            if any(x > 0 for x in people_data):
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(roads)), people_data, color=colors[:len(roads)])
                plt.title("People Affected by Congestion", fontsize=16, pad=20, weight='bold')
                plt.ylabel("Number of People", fontsize=12)
                plt.xticks(range(len(roads)), roads, rotation=45, ha='right', fontsize=10)
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(bar.get_x() + bar.get_width() / 2., height + max(people_data) * 0.01,
                                 f'{int(height):,}', ha='center', va='bottom', fontsize=10)
                plt.tight_layout()
                chart_path = temp_dir / "people.png"
                plt.savefig(chart_path, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                chart_images["people"] = str(chart_path)
                logger.info(f"Generated people bar chart at {chart_path}")

            fuel_data = [results['road_results'][road]['total_excess_fuel_l'] for road in roads]
            if any(x > 0 for x in fuel_data):
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(roads)), fuel_data, color=colors[:len(roads)])
                plt.title("Excess Fuel Consumption", fontsize=16, pad=20, weight='bold')
                plt.ylabel("Liters", fontsize=12)
                plt.xticks(range(len(roads)), roads, rotation=45, ha='right', fontsize=10)
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(bar.get_x() + bar.get_width() / 2., height + max(fuel_data) * 0.01,
                                 f'{int(height):,}', ha='center', va='bottom', fontsize=10)
                plt.tight_layout()
                chart_path = temp_dir / "fuel.png"
                plt.savefig(chart_path, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                chart_images["fuel"] = str(chart_path)
                logger.info(f"Generated fuel bar chart at {chart_path}")

            co2_data = [results['road_results'][road]['total_co2_kg'] for road in roads]
            if any(x > 0 for x in co2_data):
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(roads)), co2_data, color=colors[:len(roads)])
                plt.title("Excess CO₂ Emissions", fontsize=16, pad=20, weight='bold')
                plt.ylabel("kg", fontsize=12)
                plt.xticks(range(len(roads)), roads, rotation=45, ha='right', fontsize=10)
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(bar.get_x() + bar.get_width() / 2., height + max(co2_data) * 0.01,
                                 f'{int(height):,}', ha='center', va='bottom', fontsize=10)
                plt.tight_layout()
                chart_path = temp_dir / "co2.png"
                plt.savefig(chart_path, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                chart_images["co2"] = str(chart_path)
                logger.info(f"Generated CO2 bar chart at {chart_path}")

            cost_data = [
                results['road_results'][road]['total_fuel_cost_naira'] +
                results['road_results'][road]['total_productivity_loss_naira']
                for road in roads
            ]
            if any(x > 0 for x in cost_data):
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(roads)), cost_data, color=colors[:len(roads)])
                plt.title("Total Cost (Fuel + Productivity Loss)", fontsize=16, pad=20, weight='bold')
                plt.ylabel("Naira", fontsize=12)
                plt.xticks(range(len(roads)), roads, rotation=45, ha='right', fontsize=10)
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₦{int(x):,}'))
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(bar.get_x() + bar.get_width() / 2., height + max(cost_data) * 0.01,
                                 f'₦{int(height):,}', ha='center', va='bottom', fontsize=10)
                plt.tight_layout()
                chart_path = temp_dir / "cost.png"
                plt.savefig(chart_path, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                chart_images["cost"] = str(chart_path)
                logger.info(f"Generated cost bar chart at {chart_path}")

            return chart_images
        except Exception as e:
            logger.error(f"Error generating chart images: {str(e)}", exc_info=True)
            return {}

    def generate_pdf_report(self, output_path: str = "traffic_analysis_report.pdf",
                           homepage_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate a PDF report with homepage and analysis report using ReportLab."""
        try:
            if not output_path.endswith('.pdf'):
                output_path = output_path + '.pdf'
            output_path = Path(output_path)
            os.makedirs(output_path.parent, exist_ok=True)

            results = self.analyze_all_roads()
            if not results['road_results']:
                logger.error("No analysis results available for PDF generation")
                return None

            report_df = self.generate_report()
            if report_df.empty:
                logger.error("No report data available for PDF generation")
                return None

            vehicle_distributions = {road: self.get_vehicle_distribution(road) for road in self.road_data}
            chart_images = self.generate_chart_images()

            if homepage_data is None:
                homepage_data = {
                    'title': "Abuja Traffic Analysis System",
                    'subtitle': "Comprehensive Traffic Congestion Analysis Report",
                    'stats': [
                        {'label': 'Total Vehicles Analyzed',
                         'value': f"{results['total_summary']['total_vehicles_all_roads']:,}"},
                        {'label': 'Total People Affected',
                         'value': f"{results['total_summary']['total_people_all_roads']:,}"},
                        {'label': 'Total CO2 Emissions',
                         'value': f"{results['total_summary']['total_co2_all_roads']:,.0f} kg"},
                        {'label': 'Total Economic Impact',
                         'value': f"₦{results['total_summary']['total_fuel_cost_all_roads'] + results['total_summary']['total_productivity_loss_all_roads']:,.0f}"}
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
                        "Calculate fuel consumption and CO2 emissions",
                        "Generate comprehensive reports and visualizations"
                    ],
                    'methodology': {
                        'description': "This assessment uses vehicle-specific parameters to calculate congestion impacts:",
                        'parameters': [
                            f"Fuel Prices: ₦{self.FUEL_PRICE_PETROL:,.2f}/L (petrol), ₦{self.FUEL_PRICE_DIESEL:,.2f}/L (diesel)",
                            f"Emission Factors: {self.EMISSION_FACTOR_PETROL:,.2f} kg CO₂/L (petrol), {self.EMISSION_FACTOR_DIESEL:,.2f} kg CO₂/L (diesel)",
                            f"Productivity Value: ₦{self.VALUE_PER_MINUTE:,.2f}/minute",
                            f"Corridor Length: {self.CORRIDOR_LENGTH:,.1f} km",
                            f"Time Parameters: Free flow: 4.0 min, Congested: 45.0 min",
                            f"Advanced Parameters: Free flow speed: 60.0 km/h, Congested speed: 8.0 km/h, Acceleration: 0.5 m/s², Idle time: 30.0%, Stops per km: 2.0, Road grade: 0.0%, Temperature: 25.0°C"
                        ] if self.emission_model != EmissionModelType.BASIC else [
                            f"Fuel Prices: ₦{self.FUEL_PRICE_PETROL:,.2f}/L (petrol), ₦{self.FUEL_PRICE_DIESEL:,.2f}/L (diesel)",
                            f"Emission Factors: {self.EMISSION_FACTOR_PETROL:,.2f} kg CO₂/L (petrol), {self.EMISSION_FACTOR_DIESEL:,.2f} kg CO₂/L (diesel)",
                            f"Productivity Value: ₦{self.VALUE_PER_MINUTE:,.2f}/minute",
                            f"Corridor Length: {self.CORRIDOR_LENGTH:,.1f} km",
                            f"Time Parameters: Free flow: 4.0 min, Congested: 45.0 min"
                        ]
                    },
                    'recommendations': {
                        'key_findings': [
                            f"{max(results['road_results'].items(), key=lambda x: x[1]['total_vehicles'])[0]} shows the highest congestion impact, accounting for approximately {(max(results['road_results'].items(), key=lambda x: x[1]['total_vehicles'])[1]['total_vehicles'] / results['total_summary']['total_vehicles_all_roads'] * 100):,.1f}% of total vehicles and {(max(results['road_results'].items(), key=lambda x: x[1]['total_productivity_loss_naira'])[1]['total_productivity_loss_naira'] / results['total_summary']['total_productivity_loss_all_roads'] * 100):,.1f}% of productivity losses.",
                            f"Evening rush hour congestion affects nearly {results['total_summary']['total_people_all_roads']:,} commuters across all corridors.",
                            f"Total economic impact exceeds ₦{results['total_summary']['total_fuel_cost_all_roads'] + results['total_summary']['total_productivity_loss_all_roads']:,.0f} during the observed period.",
                            f"Environmental impact includes {results['total_summary']['total_co2_all_roads']:,.0f} kg of excess CO₂ emissions.",
                            f"Average delay per vehicle is {(results['total_summary']['total_delay_hours_all_roads'] * 60 / results['total_summary']['total_vehicles_all_roads']):,.1f} minutes across all corridors.",
                            f"Analysis performed using {self.emission_model.value.upper()} emission model for enhanced accuracy."
                        ],
                        'strategic_recommendations': [
                            f"Immediate Interventions: Implement targeted traffic management systems on {max(results['road_results'].items(), key=lambda x: x[1]['total_vehicles'])[0]}, which shows the highest congestion metrics.",
                            "Public Transportation: Enhance alternative transportation options to reduce private vehicle numbers during peak hours.",
                            "Infrastructure Investment: Prioritize road improvements based on specific vehicle type distributions observed.",
                            "Policy Measures: Consider congestion pricing or staggered work hours to distribute traffic more evenly.",
                            "Data Collection: Expand monitoring to understand daily and weekly patterns beyond current snapshot data.",
                            f"Environmental Mitigation: Implement measures to offset {results['total_summary']['total_co2_all_roads']:,.0f} kg of CO₂ emissions generated.",
                            f"Model Selection: Continue using {self.emission_model.value.upper()} model for future analyses to maintain consistency in emission calculations."
                        ]
                    }
                }
            else:
                required_keys = ['title', 'subtitle', 'stats', 'features', 'how_it_works', 'methodology',
                                'recommendations']
                if not all(key in homepage_data for key in required_keys):
                    logger.error(f"Provided homepage_data missing required keys: {required_keys}")
                    return None

            doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm,
                                    topMargin=2 * cm, bottomMargin=2 * cm)
            styles = getSampleStyleSheet()

            if not hasattr(styles, 'TitleStyle'):
                styles.add(ParagraphStyle(name='TitleStyle', fontSize=18, alignment=TA_CENTER, spaceAfter=12,
                                         textColor=colors.HexColor('#1f77b4')))

            if not hasattr(styles, 'SubtitleStyle'):
                styles.add(ParagraphStyle(name='SubtitleStyle', fontSize=14, alignment=TA_CENTER, spaceAfter=12))

            if not hasattr(styles, 'SectionTitle'):
                styles.add(ParagraphStyle(name='SectionTitle', fontSize=12, spaceBefore=12, spaceAfter=6,
                                         fontName='Helvetica-Bold'))

            if not hasattr(styles, 'Footer'):
                styles.add(ParagraphStyle(name='Footer', fontSize=8, alignment=TA_CENTER, textColor=colors.grey))

            if not hasattr(styles, 'ModelBadge'):
                styles.add(
                    ParagraphStyle(name='ModelBadge', fontSize=10, fontName='Helvetica-Bold', textColor=colors.white,
                                  backColor=colors.HexColor('#1f77b4'), spaceAfter=6, leading=12))

            elements = []
            elements.append(Paragraph(homepage_data['title'], styles['TitleStyle']))
            elements.append(Paragraph(homepage_data['subtitle'], styles['SubtitleStyle']))
            elements.append(Paragraph(
                f"Generated on {self.analysis_date} at {self.analysis_time} | Emission Model: {self.emission_model.value.capitalize()}",
                styles['Normal']))
            elements.append(Spacer(1, 0.5 * cm))

            stats_data = [[stat['label'], stat['value']] for stat in homepage_data['stats']]
            stats_table = Table(stats_data, colWidths=[8 * cm, 8 * cm])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e9ecef')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(stats_table)
            elements.append(Spacer(1, 0.5 * cm))

            elements.append(Paragraph("Features", styles['SectionTitle']))
            for feature in homepage_data['features']:
                elements.append(Paragraph(f"• {feature}", styles['Normal']))
            elements.append(Spacer(1, 0.5 * cm))

            elements.append(Paragraph("How It Works", styles['SectionTitle']))
            for step in homepage_data['how_it_works']:
                elements.append(Paragraph(f"• {step}", styles['Normal']))
            elements.append(Spacer(1, 0.5 * cm))

            elements.append(PageBreak())
            elements.append(Paragraph("Abuja Traffic Congestion Analysis Report", styles['TitleStyle']))
            elements.append(
                Paragraph("Comprehensive Analysis of Traffic Impact on Major Abuja Corridors", styles['Normal']))
            elements.append(Paragraph(f"Generated on: {self.analysis_date} at {self.analysis_time}", styles['Normal']))
            elements.append(Paragraph("Analysis Period: Evening Rush Hour (18:00)", styles['Normal']))
            elements.append(Spacer(1, 0.5 * cm))

            elements.append(Paragraph("Analysis Methodology", styles['SectionTitle']))
            model_description = {
                EmissionModelType.BASIC: "Using simple fuel consumption-based emission calculations with fixed emission factors.",
                EmissionModelType.BARTH: "Using Barth's comprehensive fuel consumption model with acceleration, deceleration, and idle time factors.",
                EmissionModelType.MOVES: "Using EPA MOVES-like emission model with speed, acceleration, and vehicle standard corrections."
            }.get(self.emission_model, f"Using {self.emission_model.value.upper()} emission calculation model.")
            elements.append(
                Paragraph(f"<b>{self.emission_model.value.upper()}</b> {model_description}", styles['ModelBadge']))
            elements.append(Spacer(1, 0.3 * cm))

            elements.append(Paragraph("Executive Summary", styles['SectionTitle']))
            elements.append(Paragraph(
                "This report provides a comprehensive analysis of traffic congestion impacts across major Abuja road corridors, quantifying economic and environmental consequences including fuel costs, productivity losses, and CO₂ emissions.",
                styles['Normal']))
            summary_data = [
                ["Total Vehicles", f"{results['total_summary']['total_vehicles_all_roads']:,}", "vehicles"],
                ["People Affected", f"{results['total_summary']['total_people_all_roads']:,}", "commuters"],
                ["Total Delay", f"{results['total_summary']['total_delay_hours_all_roads']:,.1f}", "hours"],
                ["Excess Fuel", f"{results['total_summary']['total_excess_fuel_all_roads']:,.1f}", "liters"],
                ["Fuel Cost", f"₦{results['total_summary']['total_fuel_cost_all_roads']:,.0f}", "naira"],
                ["CO₂ Emissions", f"{results['total_summary']['total_co2_all_roads']:,.0f}", "kg"]
            ]
            summary_table = Table(summary_data, colWidths=[6 * cm, 6 * cm, 4 * cm])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(summary_table)
            elements.append(Paragraph(
                f"<b>Key Insight:</b> The analysis reveals significant congestion across all corridors, with a total economic impact exceeding ₦{results['total_summary']['total_fuel_cost_all_roads'] + results['total_summary']['total_productivity_loss_all_roads']:,.0f} during the observed period.",
                styles['Normal']))
            elements.append(Spacer(1, 0.5 * cm))

            elements.append(PageBreak())
            elements.append(Paragraph("Methodology", styles['SectionTitle']))
            elements.append(Paragraph(homepage_data['methodology']['description'], styles['Normal']))
            for param in homepage_data['methodology']['parameters']:
                elements.append(Paragraph(f"• {param}", styles['Normal']))
            elements.append(Spacer(1, 0.5 * cm))

            elements.append(PageBreak())
            elements.append(Paragraph("Grand Totals Across All Roads", styles['SectionTitle']))
            totals_data = [
                ["Performance Metric", "Value", "Unit"],
                ["Total Vehicles Observed", f"{results['total_summary']['total_vehicles_all_roads']:,}", "vehicles"],
                ["Total People Affected", f"{results['total_summary']['total_people_all_roads']:,}", "people"],
                ["Total Delay Time", f"{results['total_summary']['total_delay_hours_all_roads']:,.1f}", "hours"],
                ["Total Excess Fuel Consumption", f"{results['total_summary']['total_excess_fuel_all_roads']:,.1f}",
                 "liters"],
                ["Total Fuel Cost", f"₦{results['total_summary']['total_fuel_cost_all_roads']:,.0f}", "naira"],
                ["Total CO₂ Emissions", f"{results['total_summary']['total_co2_all_roads']:,.0f}", "kg"],
                ["Total Productivity Loss", f"₦{results['total_summary']['total_productivity_loss_all_roads']:,.0f}",
                 "naira"],
                ["Total Economic Impact",
                 f"₦{results['total_summary']['total_fuel_cost_all_roads'] + results['total_summary']['total_productivity_loss_all_roads']:,.0f}",
                 "naira"]
            ]
            totals_table = Table(totals_data, colWidths=[8 * cm, 6 * cm, 2 * cm])
            totals_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e3f2fd')),
                ('FONT', (0, -1), (-1, -1), 'Helvetica-Bold', 9),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(totals_table)
            elements.append(Spacer(1, 0.5 * cm))

            elements.append(PageBreak())
            elements.append(Paragraph("Per-Road Analysis", styles['SectionTitle']))
            for road, road_data in results['road_results'].items():
                road_table_data = [
                    ["Metric", "Value"],
                    ["Vehicles", f"{road_data['total_vehicles']:,}"],
                    ["People Affected", f"{road_data['total_people']:,}"],
                    ["Excess Fuel", f"{road_data['total_excess_fuel_l']:,.1f} L"],
                    ["Fuel Cost", f"₦{road_data['total_fuel_cost_naira']:,.0f}"],
                    ["CO₂ Emissions", f"{road_data['total_co2_kg']:,.0f} kg"],
                    ["Productivity Loss", f"₦{road_data['total_productivity_loss_naira']:,.0f}"],
                    ["Total Impact",
                     f"₦{road_data['total_fuel_cost_naira'] + road_data['total_productivity_loss_naira']:,.0f}"]
                ]
                road_table = Table(road_table_data, colWidths=[8 * cm, 8 * cm])
                road_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                    ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e3f2fd')),
                    ('FONT', (0, -1), (-1, -1), 'Helvetica-Bold', 9),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                elements.append(Paragraph(road, styles['SectionTitle']))
                elements.append(road_table)
                elements.append(Spacer(1, 0.3 * cm))

            elements.append(PageBreak())
            elements.append(Paragraph("Visual Analysis", styles['SectionTitle']))
            for road in vehicle_distributions:
                elements.append(Paragraph(f"{road} Vehicle Distribution", styles['SectionTitle']))
                if f"pie_{road.replace(' ', '_').lower()}" in chart_images:
                    img_path = chart_images[f"pie_{road.replace(' ', '_').lower()}"]
                    elements.append(Image(img_path, width=16 * cm, height=12 * cm))
                for item in vehicle_distributions[road]:
                    elements.append(Paragraph(f"• {item['vehicle_type']}: {item['count']:,}", styles['Normal']))
                elements.append(Spacer(1, 0.3 * cm))
            for chart_key, chart_title in [
                ('people', 'People Affected by Congestion'),
                ('fuel', 'Excess Fuel Consumption'),
                ('co2', 'Excess CO₂ Emissions'),
                ('cost', 'Total Cost (Fuel + Productivity Loss)')
            ]:
                if chart_key in chart_images:
                    elements.append(Paragraph(chart_title, styles['SectionTitle']))
                    elements.append(Image(chart_images[chart_key], width=16 * cm, height=12 * cm))
                    elements.append(Spacer(1, 0.3 * cm))

            elements.append(PageBreak())
            elements.append(Paragraph("Detailed Report", styles['SectionTitle']))
            table_data = [
                ['Road', 'Vehicle Count', 'People Affected', 'Excess Fuel (L)', 'CO2 Emissions (kg)',
                 'Fuel Cost (Naira)', 'Productivity Loss (Naira)', 'Total Economic Impact (Naira)', 'Emission Model']
            ]
            for row in report_df.to_dict('records'):
                table_data.append([
                    row['Road'],
                    f"{row['Vehicle Count']:,}",
                    f"{row['People Affected']:,}",
                    f"{row['Excess Fuel (L)']:.1f}",
                    f"{row['CO2 Emissions (kg)']:.1f}",
                    f"₦{row['Fuel Cost (Naira)']:,.0f}",
                    f"₦{row['Productivity Loss (Naira)']:,.0f}",
                    f"₦{row['Total Economic Impact (Naira)']:,.0f}",
                    row['Emission Model']
                ])
            results_table = Table(table_data,
                                 colWidths=[3 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 3 * cm, 2 * cm])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-2, -1), 'RIGHT'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(results_table)
            elements.append(Spacer(1, 0.5 * cm))

            elements.append(PageBreak())
            elements.append(Paragraph("Conclusions & Recommendations", styles['SectionTitle']))
            elements.append(Paragraph("Key Findings", styles['SectionTitle']))
            for finding in homepage_data['recommendations']['key_findings']:
                elements.append(Paragraph(f"• {finding}", styles['Normal']))
            elements.append(Spacer(1, 0.3 * cm))
            elements.append(Paragraph("Strategic Recommendations", styles['SectionTitle']))
            for recommendation in homepage_data['recommendations']['strategic_recommendations']:
                elements.append(Paragraph(f"• {recommendation}", styles['Normal']))
            elements.append(Spacer(1, 0.5 * cm))

            elements.append(Paragraph(
                f"Generated on {self.analysis_date} at {self.analysis_time} by Abuja Traffic Analysis System<br/>&copy; {datetime.now().year} | Powered by Opygoal Technology Ltd | Developed by Oladotun Ajakaiye",
                styles['Footer']
            ))

            doc.build(elements)
            logger.info(f"PDF report generated successfully at {output_path}")

            temp_dir = Path('temp_charts')
            if temp_dir.exists():
                for chart_file in temp_dir.glob("*.png"):
                    try:
                        chart_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary chart file {chart_file}: {str(e)}")
                try:
                    temp_dir.rmdir()
                except Exception as e:
                    logger.warning(f"Failed to delete temporary directory {temp_dir}: {str(e)}")

            return str(output_path)
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
            return None

if __name__ == "__main__":
    for model_type in ['basic', 'barth', 'moves']:
        model = TrafficAnalysisModel(emission_model=EmissionModelType(model_type.upper()))
        pdf_path = model.generate_pdf_report(output_path=f"traffic_analysis_report_{model_type}.pdf")
        if pdf_path:
            print(f"PDF report for {model_type} model generated at: {pdf_path}")
        else:
            print(f"Failed to generate PDF report for {model_type} model")