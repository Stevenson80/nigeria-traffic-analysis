#!/usr/bin/env python3
"""
TrafficAnalysisModel class for analyzing traffic data with individual metric calculations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import os
import math  # Added import for Barth polynomial
from enum import Enum
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_analysis_models.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Define Emission Model Types
class EmissionModelType(Enum):
    BASIC = "basic"
    BARTH = "barth"
    MOVES = "moves"

    @property
    def display_name(self):
        """Return a human-readable name for the emission model."""
        return {
            "basic": "Basic Model",
            "barth": "Barth Model",
            "moves": "MOVES Model"
        }[self.value]


# Define Valid Vehicle Types
valid_vehicle_types = ['Motorcycles', 'Cars/SUVs', 'Coasters/Buses', 'Trucks', 'Tankers/Trailers']

# Define Predefined Nigerian States
NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa",
    "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo",
    "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna",
    "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos",
    "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo",
    "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara",
    "FCT Abuja"
]


# Define Calculation Metrics
class CalculationMetric(Enum):
    PRODUCTIVITY_LOSS = "productivity_loss"
    EXCESS_FUEL = "excess_fuel"
    CO2_EMISSIONS = "co2_emissions"


class TrafficAnalysisModel:
    # Constants (used only as defaults if not provided in data)
    EMISSION_FACTOR_PETROL = 2.31  # kg CO₂/liter
    EMISSION_FACTOR_DIESEL = 2.68  # kg CO₂/liter
    VALUE_PER_MINUTE = 8.8  # Naira/minute (default, should be overridden by user input)
    CORRIDOR_LENGTH = 6.0  # km (default, should be overridden by user input)
    BARTH_IDLE_FUEL_CONSUMPTION_PETROL = 0.6  # liters/hour
    BARTH_IDLE_FUEL_CONSUMPTION_DIESEL = 0.8  # liters/hour
    BARTH_ACCEL_FACTOR_PETROL = 0.00035
    BARTH_ACCEL_FACTOR_DIESEL = 0.00045
    MOVES_BASE_EMISSION_RATE_PETROL = 2.0  # g/km
    MOVES_BASE_EMISSION_RATE_DIESEL = 2.5  # g/km
    MOVES_SPEED_CORRECTION_FACTOR = 0.08
    MOVES_ACCEL_CORRECTION_FACTOR = 0.15
    # Default fuel prices reflecting current Nigerian market (Oct 2024)
    DEFAULT_FUEL_PRICE_PETROL = 850.0  # ₦ per liter - minimum current price
    DEFAULT_FUEL_PRICE_DIESEL = 1000.0  # ₦ per liter - minimum current price
    MAX_CITIES_PER_STATE = 20
    MAX_STATES = 37

    # CORRECTED Barth coefficients
    BARTH_COEFFICIENTS = {
        'alpha': 0.00015,  # Engine displacement coefficient
        'beta': 0.0008,  # Weight coefficient
        'gamma': 0.0005,  # Road grade coefficient
        'traffic_flow': 0.001,  # Speed effect coefficient
        'road_gradient': 0.0005,  # Road grade effect
        'acceleration': 0.0003  # Acceleration effect
    }

    def __init__(self, emission_model: EmissionModelType = EmissionModelType.BASIC):
        """Initialize the TrafficAnalysisModel with emission model."""
        self.vehicle_parameters = {
            'Motorcycles': {
                'fuel_type': 'petrol',
                'fuel_efficiency_free_flow': 0.08,
                'fuel_efficiency_congested': 0.12,
                'occupancy_avg': 1.2,
                'emission_factor': 0.1,
                'weight_kg': 150,
                'engine_displacement_cc': 125
            },
            'Cars/SUVs': {
                'fuel_type': 'petrol',
                'fuel_efficiency_free_flow': 0.10,
                'fuel_efficiency_congested': 0.15,
                'occupancy_avg': 2.5,
                'emission_factor': 0.2,
                'weight_kg': 1500,
                'engine_displacement_cc': 2000
            },
            'Coasters/Buses': {
                'fuel_type': 'diesel',
                'fuel_efficiency_free_flow': 0.30,
                'fuel_efficiency_congested': 0.45,
                'occupancy_avg': 15.0,
                'emission_factor': 0.4,
                'weight_kg': 6000,
                'engine_displacement_cc': 5000
            },
            'Trucks': {
                'fuel_type': 'diesel',
                'fuel_efficiency_free_flow': 0.35,
                'fuel_efficiency_congested': 0.50,
                'occupancy_avg': 2.0,
                'emission_factor': 0.5,
                'weight_kg': 8000,
                'engine_displacement_cc': 6000
            },
            'Tankers/Trailers': {
                'fuel_type': 'diesel',
                'fuel_efficiency_free_flow': 0.40,
                'fuel_efficiency_congested': 0.60,
                'occupancy_avg': 1.5,
                'emission_factor': 0.7,
                'weight_kg': 12000,
                'engine_displacement_cc': 8000
            }
        }
        self.emission_model = emission_model
        if not isinstance(self.emission_model, EmissionModelType):
            raise ValueError(f"Invalid emission model: {self.emission_model}")
        self.state_city_data = {}
        self.data = pd.DataFrame()
        self.analysis_date = datetime.now().strftime('%Y-%m-%d')
        self.analysis_time = datetime.now().strftime('%H:%M')
        self.load_default_data()

    def load_default_data(self):
        """Load default (empty) data structure."""
        try:
            self.state_city_data = {}
            self.data = pd.DataFrame(columns=[
                'Date', 'Time', 'State', 'City', 'Road', 'Vehicle_Type', 'Real_Vehicle_Count', 'Real_VOR',
                'Congested_Travel_Time_Minutes', 'Distance_KM', 'Free_Flow_Time_Minutes',
                'Value_Per_Minute_Naira', 'Fuel_Cost_Per_Liter_Petrol', 'Fuel_Cost_Per_Liter_Diesel',
                'Emission_Factor_Petrol', 'Emission_Factor_Diesel', 'Real_Delay_Time',
                'Free_Flow_Speed_KPH', 'Congested_Speed_KPH', 'Avg_Acceleration', 'Avg_Deceleration',
                'Idle_Time_Percentage', 'Stops_Per_KM', 'Road_Grade', 'Temperature_C',
                'Emission_Model', 'Barth_Alpha', 'Barth_Beta', 'Barth_Gamma',
                'Barth_Traffic_Flow', 'Barth_Road_Gradient', 'Barth_Acceleration',
                'Vehicle_Emission_Factor', 'flow_rate_veh_per_hr', 'analysis_period_hr', 'class_distribution'
            ])
            logger.info("Initialized empty data structure")
        except Exception as e:
            logger.error(f"Error initializing default data: {str(e)}", exc_info=True)
            self.data = pd.DataFrame()
            self.state_city_data = {}

    def add_city_data(self, state_name: str, city_name: str, vehicle_data: list):
        """Add data for a specific city within a state."""
        try:
            # Validate state
            if state_name not in NIGERIAN_STATES:
                logger.error(f"Invalid state: {state_name}. Must be one of: {NIGERIAN_STATES}")
                raise ValueError(f"Invalid state: {state_name}. Must be one of: {NIGERIAN_STATES}")

            # Check state limit
            if state_name not in self.state_city_data and len(self.state_city_data) >= self.MAX_STATES:
                logger.error(f"Maximum number of states ({self.MAX_STATES}) reached")
                raise ValueError(f"Cannot add more than {self.MAX_STATES} states")

            # Initialize state if not present
            if state_name not in self.state_city_data:
                self.state_city_data[state_name] = {}

            # Check city limit per state
            if len(self.state_city_data[state_name]) >= self.MAX_CITIES_PER_STATE:
                logger.error(f"Maximum number of cities ({self.MAX_CITIES_PER_STATE}) reached for state {state_name}")
                raise ValueError(f"Cannot add more than {self.MAX_CITIES_PER_STATE} cities in {state_name}")

            df_city = pd.DataFrame(vehicle_data)
            df_city['State'] = state_name
            df_city['City'] = city_name
            df_city['Date'] = self.analysis_date
            df_city['Time'] = self.analysis_time

            # Validate vehicle types
            invalid_vehicles = df_city[~df_city['Vehicle_Type'].isin(valid_vehicle_types)]['Vehicle_Type'].unique()
            if invalid_vehicles:
                logger.error(f"Invalid vehicle types: {invalid_vehicles}")
                raise ValueError(f"Invalid vehicle types: {invalid_vehicles}")

            # Validate non-zero vehicle counts
            if not any(float(row.get('Real_Vehicle_Count', 0)) > 0 for _, row in df_city.iterrows()):
                logger.error(f"No valid vehicle counts for {city_name}, {state_name}")
                raise ValueError(f"No valid vehicle counts for {city_name}, {state_name}")

            # Validate positive occupancy values
            if any(float(row.get('Real_VOR', 0)) <= 0 for _, row in df_city.iterrows() if
                   float(row.get('Real_Vehicle_Count', 0)) > 0):
                logger.error(f"Invalid occupancy values for {city_name}, {state_name}")
                raise ValueError(f"Invalid occupancy values for {city_name}, {state_name}")

            # VALIDATION: Ensure required fields exist
            required_fields = ['Congested_Travel_Time_Minutes', 'Free_Flow_Time_Minutes', 'Distance_KM']
            for field in required_fields:
                if field not in df_city.columns or df_city[field].isna().all():
                    logger.error(f"Missing required field: {field} for {city_name}, {state_name}")
                    raise ValueError(f"Required field '{field}' is missing for {city_name}")

            # Only set defaults for non-essential advanced fields that don't affect core calculations
            advanced_defaults = {
                'Free_Flow_Speed_KPH': 60.0,
                'Congested_Speed_KPH': 8.0,
                'Avg_Acceleration': 0.5,
                'Avg_Deceleration': 0.5,
                'Idle_Time_Percentage': 0.3,
                'Stops_Per_KM': 2.0,
                'Road_Grade': 0.0,
                'Temperature_C': 25.0,
            }

            for col, default in advanced_defaults.items():
                if col not in df_city.columns or df_city[col].isna().all():
                    df_city[col] = default
                    logger.info(f"Set advanced field {col} to default: {default}")

            # Set default fuel prices if not provided by user
            if 'Fuel_Cost_Per_Liter_Petrol' not in df_city.columns or df_city[
                'Fuel_Cost_Per_Liter_Petrol'].isna().all():
                df_city['Fuel_Cost_Per_Liter_Petrol'] = self.DEFAULT_FUEL_PRICE_PETROL
                logger.info(f"Using default petrol price: ₦{self.DEFAULT_FUEL_PRICE_PETROL:,.2f}/L")

            if 'Fuel_Cost_Per_Liter_Diesel' not in df_city.columns or df_city[
                'Fuel_Cost_Per_Liter_Diesel'].isna().all():
                df_city['Fuel_Cost_Per_Liter_Diesel'] = self.DEFAULT_FUEL_PRICE_DIESEL
                logger.info(f"Using default diesel price: ₦{self.DEFAULT_FUEL_PRICE_DIESEL:,.2f}/L")

            # Enhanced idle time validation
            if 'Idle_Time_Percentage' in df_city.columns:
                df_city['Idle_Time_Percentage'] = df_city['Idle_Time_Percentage'].apply(
                    lambda x: max(0.0, min(100.0, float(x))) if pd.notna(x) else 0.3
                )
                df_city['Idle_Time_Percentage'] = df_city['Idle_Time_Percentage'].apply(
                    lambda x: x / 100.0 if x > 1.0 else x
                )
                df_city['Idle_Time_Percentage'] = df_city.apply(
                    lambda row: min(row['Idle_Time_Percentage'],
                                    float(row['Congested_Travel_Time_Minutes']) / 60.0)
                    if pd.notna(row['Congested_Travel_Time_Minutes']) else row['Idle_Time_Percentage'],
                    axis=1
                )

            # Calculate Real_Delay_Time
            if df_city['Real_Delay_Time'].isna().all() or df_city['Real_Delay_Time'].iloc[0] is None:
                df_city['Real_Delay_Time'] = df_city['Congested_Travel_Time_Minutes'] - df_city[
                    'Free_Flow_Time_Minutes']

            # Validate fuel prices
            if 'Fuel_Cost_Per_Liter_Petrol' not in df_city.columns or df_city[
                'Fuel_Cost_Per_Liter_Petrol'].isna().all() or float(df_city['Fuel_Cost_Per_Liter_Petrol'].iloc[0]) <= 0:
                logger.error("Fuel_Cost_Per_Liter_Petrol must be provided and greater than zero")
                raise ValueError("Fuel_Cost_Per_Liter_Petrol must be provided and greater than zero")
            if 'Fuel_Cost_Per_Liter_Diesel' not in df_city.columns or df_city[
                'Fuel_Cost_Per_Liter_Diesel'].isna().all() or float(df_city['Fuel_Cost_Per_Liter_Diesel'].iloc[0]) <= 0:
                logger.error("Fuel_Cost_Per_Liter_Diesel must be provided and greater than zero")
                raise ValueError("Fuel_Cost_Per_Liter_Diesel must be provided and greater than zero")

            # Log the fuel prices being used
            petrol_price = float(df_city['Fuel_Cost_Per_Liter_Petrol'].iloc[0])
            diesel_price = float(df_city['Fuel_Cost_Per_Liter_Diesel'].iloc[0])
            logger.info(f"Using fuel prices - Petrol: ₦{petrol_price:,.2f}/L, Diesel: ₦{diesel_price:,.2f}/L")

            # Set vehicle-specific emission factors
            df_city['Vehicle_Emission_Factor'] = df_city.apply(
                lambda row: float(row.get('Vehicle_Emission_Factor',
                                          self.vehicle_parameters.get(row['Vehicle_Type'], {}).get('emission_factor',
                                                                                                   0.2))),
                axis=1
            )

            # Add to main data
            if self.data.empty:
                self.data = df_city
            else:
                self.data = pd.concat([self.data, df_city], ignore_index=True)

            # Update state_city_data
            self.state_city_data[state_name][city_name] = df_city
            logger.info(f"Added data for city: {city_name}, state: {state_name}")

        except Exception as e:
            logger.error(f"Error adding city data for {city_name}, {state_name}: {str(e)}", exc_info=True)
            raise

    def remove_city_data(self, state_name: str, city_name: str):
        """Remove data for a specific city within a state."""
        try:
            if state_name in self.state_city_data and city_name in self.state_city_data[state_name]:
                del self.state_city_data[state_name][city_name]
                if not self.state_city_data[state_name]:
                    del self.state_city_data[state_name]
                self.data = self.data[(self.data['State'] != state_name) | (self.data['City'] != city_name)]
                logger.info(f"Removed data for city: {city_name}, state: {state_name}")
            else:
                logger.warning(f"City {city_name} in state {state_name} not found in data")
        except Exception as e:
            logger.error(f"Error removing city data for {city_name}, {state_name}: {str(e)}", exc_info=True)
            raise

    def validate_model_selection(self, metric: CalculationMetric, model: EmissionModelType):
        """Validate that the selected model is appropriate for the metric."""
        valid_combinations = {
            CalculationMetric.PRODUCTIVITY_LOSS: [EmissionModelType.BASIC, EmissionModelType.BARTH,
                                                  EmissionModelType.MOVES],
            CalculationMetric.EXCESS_FUEL: [EmissionModelType.BARTH, EmissionModelType.MOVES],
            CalculationMetric.CO2_EMISSIONS: [EmissionModelType.BARTH, EmissionModelType.MOVES]
        }

        if model not in valid_combinations.get(metric, []):
            logger.warning(f"Model {model.value} may not be optimal for metric {metric.value}. "
                           f"Recommended models: {[m.value for m in valid_combinations[metric]]}")
            return False
        return True

    def get_intermediate_calculations(self, row, metric: CalculationMetric, model: EmissionModelType):
        """Return intermediate calculations for debugging and verification."""
        calculations = {}

        try:
            # FIXED: Use RAW vehicle count without analysis period multiplication
            vehicle_count = float(row['Real_Vehicle_Count'])
            occupancy = float(
                row.get('Real_VOR', self.vehicle_parameters.get(row['Vehicle_Type'], {}).get('occupancy_avg', 1.0)))
            distance_km = float(row['Distance_KM'])
            delay_minutes = float(
                row.get('Real_Delay_Time', row['Congested_Travel_Time_Minutes'] - row['Free_Flow_Time_Minutes']))

            calculations['basic_parameters'] = {
                'vehicle_count': vehicle_count,
                'occupancy': occupancy,
                'distance_km': distance_km,
                'delay_minutes': delay_minutes,
                'fuel_price_petrol': float(row.get('Fuel_Cost_Per_Liter_Petrol', self.DEFAULT_FUEL_PRICE_PETROL)),
                'fuel_price_diesel': float(row.get('Fuel_Cost_Per_Liter_Diesel', self.DEFAULT_FUEL_PRICE_DIESEL)),
                'selected_model': model.value,
                'analysis_period_hr': float(row.get('analysis_period_hr', 1.0))
            }

            if metric == CalculationMetric.PRODUCTIVITY_LOSS:
                value_per_minute = float(row.get('Value_Per_Minute_Naira', self.VALUE_PER_MINUTE))
                calculations['productivity_details'] = {
                    'value_per_minute': value_per_minute,
                    'total_people': vehicle_count * occupancy,
                    'total_delay_minutes': delay_minutes,
                    'calculated_loss': vehicle_count * occupancy * delay_minutes * value_per_minute
                }

            elif metric == CalculationMetric.EXCESS_FUEL:
                if model == EmissionModelType.BARTH:
                    free_flow_fuel = self.calculate_barth_fuel_consumption_single(
                        speed=float(row.get('Free_Flow_Speed_KPH', 60.0)),
                        vehicle_type=row['Vehicle_Type'],
                        distance=distance_km,
                        count=vehicle_count,
                        road_grade=float(row.get('Road_Grade', 0.0)),
                        is_congested=False,
                        stops_per_km=0.1,
                        idle_time=0
                    )

                    congested_fuel = self.calculate_barth_fuel_consumption_single(
                        speed=float(row.get('Congested_Speed_KPH', 8.0)),
                        vehicle_type=row['Vehicle_Type'],
                        distance=distance_km,
                        count=vehicle_count,
                        road_grade=float(row.get('Road_Grade', 0.0)),
                        is_congested=True,
                        stops_per_km=float(row.get('Stops_Per_KM', 2.0)),
                        idle_time=float(row.get('Idle_Time_Percentage', 0.3)) * float(
                            row.get('Congested_Travel_Time_Minutes', 45.0))
                    )
                else:  # MOVES model
                    free_flow_fuel = self.calculate_moves_fuel_consumption_single(
                        speed=float(row.get('Free_Flow_Speed_KPH', 60.0)),
                        stops_per_km=0.5,
                        vehicle_type=row['Vehicle_Type'],
                        distance=distance_km,
                        count=vehicle_count,
                        road_grade=float(row.get('Road_Grade', 0.0)),
                        temperature_c=float(row.get('Temperature_C', 25.0)),
                        is_free_flow=True
                    )
                    congested_fuel = self.calculate_moves_fuel_consumption_single(
                        speed=float(row.get('Congested_Speed_KPH', 8.0)),
                        stops_per_km=float(row.get('Stops_Per_KM', 2.0)),
                        vehicle_type=row['Vehicle_Type'],
                        distance=distance_km,
                        count=vehicle_count,
                        road_grade=float(row.get('Road_Grade', 0.0)),
                        temperature_c=float(row.get('Temperature_C', 25.0)),
                        is_free_flow=False
                    )

                fuel_type = self.vehicle_parameters.get(row['Vehicle_Type'], {}).get('fuel_type', 'petrol')
                fuel_price = float(row.get('Fuel_Cost_Per_Liter_Petrol',
                                           self.DEFAULT_FUEL_PRICE_PETROL)) if fuel_type == 'petrol' else float(
                    row.get('Fuel_Cost_Per_Liter_Diesel', self.DEFAULT_FUEL_PRICE_DIESEL))

                excess_fuel_liters = max(0.0, congested_fuel - free_flow_fuel)
                excess_fuel_cost = excess_fuel_liters * fuel_price

                calculations['fuel_details'] = {
                    'free_flow_fuel_liters': free_flow_fuel,
                    'congested_fuel_liters': congested_fuel,
                    'excess_fuel_liters': excess_fuel_liters,
                    'fuel_type': fuel_type,
                    'fuel_price_per_liter': fuel_price,
                    'excess_fuel_cost': excess_fuel_cost,
                    'model_used': model.value
                }

            elif metric == CalculationMetric.CO2_EMISSIONS:
                if model == EmissionModelType.BARTH:
                    total_fuel = self.calculate_barth_fuel_consumption_single(
                        speed=float(row.get('Congested_Speed_KPH', 8.0)),
                        vehicle_type=row['Vehicle_Type'],
                        distance=distance_km,
                        count=vehicle_count,
                        road_grade=float(row.get('Road_Grade', 0.0)),
                        is_congested=True,
                        stops_per_km=float(row.get('Stops_Per_KM', 2.0)),
                        idle_time=float(row.get('Idle_Time_Percentage', 0.3)) * float(
                            row.get('Congested_Travel_Time_Minutes', 45.0))
                    )
                elif model == EmissionModelType.MOVES:
                    total_fuel = self.calculate_moves_fuel_consumption_single(
                        speed=float(row.get('Congested_Speed_KPH', 8.0)),
                        stops_per_km=float(row.get('Stops_Per_KM', 2.0)),
                        vehicle_type=row['Vehicle_Type'],
                        distance=distance_km,
                        count=vehicle_count,
                        road_grade=float(row.get('Road_Grade', 0.0)),
                        temperature_c=float(row.get('Temperature_C', 25.0)),
                        is_free_flow=False
                    )
                else:  # BASIC model
                    vehicle_params = self.vehicle_parameters.get(row['Vehicle_Type'], {})
                    fuel_efficiency_congested = vehicle_params.get('fuel_efficiency_congested', 0.15)
                    total_fuel = vehicle_count * distance_km * fuel_efficiency_congested

                fuel_type = self.vehicle_parameters.get(row['Vehicle_Type'], {}).get('fuel_type', 'petrol')
                emission_factor = float(
                    row.get('Emission_Factor_Petrol', self.EMISSION_FACTOR_PETROL)) if fuel_type == 'petrol' else float(
                    row.get('Emission_Factor_Diesel', self.EMISSION_FACTOR_DIESEL))

                calculations['emission_details'] = {
                    'total_fuel_consumption_liters': total_fuel,
                    'fuel_type': fuel_type,
                    'emission_factor_kg_per_liter': emission_factor,
                    'total_emissions_kg': total_fuel * emission_factor,
                    'model_used': model.value
                }

        except Exception as e:
            logger.error(f"Error in intermediate calculations: {str(e)}")
            calculations['error'] = str(e)

        return calculations

    def calculate_metric(self, metric: CalculationMetric, model: EmissionModelType = None) -> dict:
        """Calculate the specified metric using the selected model with per-vehicle breakdowns."""
        logger.debug(f"Calculating metric {metric} with model {model or self.emission_model}")
        if model is None:
            model = self.emission_model
        if not isinstance(model, EmissionModelType):
            raise ValueError(f"Invalid emission model: {model}")

        # Validate model selection
        if not self.validate_model_selection(metric, model):
            logger.warning(f"Using model {model.value} for {metric.value} despite validation warning")

        try:
            results = {
                'state_results': {},
                'city_results': {},
                'total_summary': {}
            }
            total_vehicles = 0
            total_people = 0
            total_metric_value = 0

            for state, cities in self.state_city_data.items():
                state_total = {'total_vehicles': 0, 'total_people': 0, f'total_{metric.value}': 0}
                results['city_results'][state] = {}

                for city, df in cities.items():
                    # FIXED: Added per-vehicle-type metrics storage
                    city_result = {
                        'total_vehicles': 0,
                        'total_people': 0,
                        f'total_{metric.value}': 0,
                        'vehicle_breakdown': {},
                        'occupancy_rates': {},
                        f'{metric.value}_by_vehicle_type': {}  # NEW: Store per-vehicle metrics
                    }

                    for _, row in df.iterrows():
                        # FIXED: Use RAW vehicle count without analysis period multiplication
                        vehicle_count = float(row['Real_Vehicle_Count'])
                        occupancy = float(row.get('Real_VOR', self.vehicle_parameters.get(row['Vehicle_Type'], {}).get(
                            'occupancy_avg', 1.0)))
                        vehicle_type = row['Vehicle_Type']

                        city_result['total_vehicles'] += vehicle_count
                        city_result['total_people'] += vehicle_count * occupancy
                        city_result['vehicle_breakdown'][vehicle_type] = vehicle_count
                        city_result['occupancy_rates'][vehicle_type] = occupancy  # Store actual occupancy

                        # Calculate metric for this specific vehicle type
                        if metric == CalculationMetric.PRODUCTIVITY_LOSS:
                            vehicle_metric_value = self.calculate_productivity_loss(row)
                            city_result['total_productivity_loss'] += vehicle_metric_value
                            city_result['productivity_loss_by_vehicle_type'][vehicle_type] = vehicle_metric_value
                        elif metric == CalculationMetric.EXCESS_FUEL:
                            vehicle_metric_value = self.calculate_excess_fuel(row, model)
                            city_result['total_excess_fuel'] += vehicle_metric_value
                            city_result['excess_fuel_by_vehicle_type'][vehicle_type] = vehicle_metric_value
                        elif metric == CalculationMetric.CO2_EMISSIONS:
                            vehicle_metric_value = self.calculate_co2_emissions(row, model)
                            city_result['total_co2_emissions'] += vehicle_metric_value
                            city_result['co2_emissions_by_vehicle_type'][vehicle_type] = vehicle_metric_value

                    results['city_results'][state][city] = city_result
                    state_total['total_vehicles'] += city_result['total_vehicles']
                    state_total['total_people'] += city_result['total_people']
                    state_total[f'total_{metric.value}'] += city_result[f'total_{metric.value}']

                results['state_results'][state] = state_total
                total_vehicles += state_total['total_vehicles']
                total_people += state_total['total_people']
                total_metric_value += state_total[f'total_{metric.value}']

            results['total_summary'] = {
                'total_vehicles_all_cities': total_vehicles,
                'total_people_all_cities': total_people,
                f'total_{metric.value}_all_cities': total_metric_value
            }
            return results
        except Exception as e:
            logger.error(f"Error calculating metric {metric}: {str(e)}", exc_info=True)
            return {'state_results': {}, 'city_results': {}, 'total_summary': {}}

    def calculate_productivity_loss(self, row):
        """Calculate productivity loss for a given row."""
        try:
            # FIXED: Use RAW vehicle count without analysis period multiplication
            vehicle_count = float(row['Real_Vehicle_Count'])
            occupancy = float(
                row.get('Real_VOR', self.vehicle_parameters.get(row['Vehicle_Type'], {}).get('occupancy_avg', 1.0)))
            delay_minutes = float(
                row.get('Real_Delay_Time', row['Congested_Travel_Time_Minutes'] - row['Free_Flow_Time_Minutes']))
            value_per_minute = float(row.get('Value_Per_Minute_Naira', self.VALUE_PER_MINUTE))
            if value_per_minute <= 0:
                logger.warning(f"Invalid value per minute in {row['City']}, {row['State']}: {value_per_minute}")
                raise ValueError("Value per minute must be greater than zero")
            return vehicle_count * occupancy * delay_minutes * value_per_minute
        except (ValueError, KeyError) as e:
            logger.error(f"Error in calculate_productivity_loss: {str(e)}")
            return 0.0

    def calculate_excess_fuel(self, row, model: EmissionModelType):
        """Calculate excess fuel cost for a given row."""
        try:
            # FIXED: Use RAW vehicle count without analysis period multiplication
            vehicle_count = float(row['Real_Vehicle_Count'])
            distance_km = float(row['Distance_KM'])
            vehicle_type = row['Vehicle_Type']

            free_flow_speed = float(row.get('Free_Flow_Speed_KPH', 60.0))
            congested_speed = float(row.get('Congested_Speed_KPH', 8.0))

            if model == EmissionModelType.BARTH:
                stops_per_km = float(row.get('Stops_Per_KM', 2.0))
                idle_time_pct = float(row.get('Idle_Time_Percentage', 0.3))

                # FREE FLOW fuel (smooth traffic)
                free_flow_fuel = self.calculate_barth_fuel_consumption_single(
                    speed=free_flow_speed,
                    vehicle_type=vehicle_type,
                    distance=distance_km,
                    count=vehicle_count,
                    road_grade=float(row.get('Road_Grade', 0.0)),
                    is_congested=False,  # NOT congested
                    stops_per_km=0.1,  # Few stops in free flow
                    idle_time=0  # No idle time in free flow
                )

                # CONGESTED fuel (traffic jam)
                congested_fuel = self.calculate_barth_fuel_consumption_single(
                    speed=congested_speed,
                    vehicle_type=vehicle_type,
                    distance=distance_km,
                    count=vehicle_count,
                    road_grade=float(row.get('Road_Grade', 0.0)),
                    is_congested=True,  # IS congested
                    stops_per_km=stops_per_km,
                    idle_time=idle_time_pct * float(row.get('Congested_Travel_Time_Minutes', 45.0))
                )

            else:  # MOVES model (keep existing)
                free_flow_fuel = self.calculate_moves_fuel_consumption_single(
                    speed=free_flow_speed,
                    stops_per_km=0.5,
                    vehicle_type=vehicle_type,
                    distance=distance_km,
                    count=vehicle_count,
                    road_grade=float(row.get('Road_Grade', 0.0)),
                    temperature_c=float(row.get('Temperature_C', 25.0)),
                    is_free_flow=True
                )
                congested_fuel = self.calculate_moves_fuel_consumption_single(
                    speed=congested_speed,
                    stops_per_km=float(row.get('Stops_Per_KM', 2.0)),
                    vehicle_type=vehicle_type,
                    distance=distance_km,
                    count=vehicle_count,
                    road_grade=float(row.get('Road_Grade', 0.0)),
                    temperature_c=float(row.get('Temperature_C', 25.0)),
                    is_free_flow=False
                )

            excess_fuel_liters = max(0.0, congested_fuel - free_flow_fuel)

            # FIXED: Add validation for excess fuel calculation
            excess_fuel_liters = self.validate_fuel_consumption(
                excess_fuel_liters, vehicle_type, distance_km, vehicle_count
            )

            # Get fuel type and price for cost calculation
            fuel_type = self.vehicle_parameters.get(vehicle_type, {}).get('fuel_type', 'petrol')
            fuel_price = float(row.get('Fuel_Cost_Per_Liter_Petrol',
                                       self.DEFAULT_FUEL_PRICE_PETROL)) if fuel_type == 'petrol' else float(
                row.get('Fuel_Cost_Per_Liter_Diesel', self.DEFAULT_FUEL_PRICE_DIESEL))

            # Calculate cost - FIXED: Return just the cost value, not a dictionary
            excess_fuel_cost = excess_fuel_liters * fuel_price

            logger.info(
                f"Excess fuel for {vehicle_type}: {free_flow_fuel:.1f}L (free) -> {congested_fuel:.1f}L (congested) = {excess_fuel_liters:.1f}L excess = ₦{excess_fuel_cost:,.0f}")

            # FIXED: Return just the cost value, not a dictionary
            return excess_fuel_cost

        except (ValueError, KeyError) as e:
            logger.error(f"Error in calculate_excess_fuel: {str(e)}")
            return 0.0

    def calculate_barth_fuel_consumption_single(self, speed: float, vehicle_type: str, distance: float,
                                                count: int, road_grade: float, is_congested: bool,
                                                stops_per_km: float = 0, idle_time: float = 0):
        """FIXED Barth fuel consumption calculation with PROPER idle time handling"""
        vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
        if not vehicle_params:
            return 0.0

        # REALISTIC base fuel consumption (liters/km) - ACTUAL REAL-WORLD VALUES
        base_consumption_rates = {
            'Motorcycles': 0.03,  # 33 km/L - REALISTIC
            'Cars/SUVs': 0.08,  # 12.5 km/L - REALISTIC
            'Coasters/Buses': 0.25,  # 4 km/L - REALISTIC
            'Trucks': 0.30,  # 3.3 km/L - REALISTIC
            'Tankers/Trailers': 0.35  # 2.85 km/L - REALISTIC
        }

        base_consumption = base_consumption_rates.get(vehicle_type, 0.1)

        # Speed effect - REALISTIC adjustments
        speed_effect = 0.0
        if speed < 20.0:  # Very congested - 40% increase
            speed_effect = base_consumption * 0.4
        elif speed < 40.0:  # Congested - 20% increase
            speed_effect = base_consumption * 0.2
        elif speed > 80.0:  # High speed - 30% increase
            speed_effect = base_consumption * 0.3

        # Congestion effects - SMALL realistic adjustments
        congestion_effect = 0.0
        if is_congested:
            congestion_effect = stops_per_km * 0.002  # Very small effect per stop
            # Road grade effect (minor)
            grade_effect = abs(road_grade) * 0.005

        # Total consumption per km - ENSURE REALISTIC BOUNDS
        consumption_per_km = base_consumption + speed_effect + congestion_effect

        # Apply realistic bounds (never less than 50% or more than 200% of base)
        min_consumption = base_consumption * 0.5
        max_consumption = base_consumption * 2.0
        consumption_per_km = max(min_consumption, min(max_consumption, consumption_per_km))

        # CRITICAL FIX: Calculate MOVING fuel consumption only
        moving_fuel = consumption_per_km * distance * count

        # Calculate IDLE fuel consumption separately
        idle_fuel = 0.0
        if is_congested and idle_time > 0:
            # Realistic idle fuel consumption (liters/hour) - ACTUAL VALUES
            idle_rates = {
                'Motorcycles': 0.2,  # 200 ml/hour
                'Cars/SUVs': 0.4,  # 400 ml/hour
                'Coasters/Buses': 0.8,  # 800 ml/hour
                'Trucks': 1.0,  # 1.0 L/hour
                'Tankers/Trailers': 1.2  # 1.2 L/hour
            }
            idle_rate = idle_rates.get(vehicle_type, 0.5)
            idle_fuel = (idle_time / 60.0) * idle_rate * count

        # CRITICAL FIX: Total fuel = moving fuel + idle fuel
        total_fuel = moving_fuel + idle_fuel

        # FINAL VALIDATION: Ensure total fuel is realistic
        max_reasonable_fuel = base_consumption * distance * count * 3.0  # Allow 3x buffer
        if total_fuel > max_reasonable_fuel:
            logger.warning(
                f"Unrealistic fuel {total_fuel:.1f}L for {vehicle_type}, capping to {max_reasonable_fuel:.1f}L")
            return max_reasonable_fuel

        logger.debug(
            f"Fuel for {vehicle_type}: {moving_fuel:.1f}L moving + {idle_fuel:.1f}L idle = {total_fuel:.1f}L total")
        return total_fuel

    def calculate_moves_fuel_consumption_single(self, speed: float, stops_per_km: float, vehicle_type: str,
                                                distance: float, count: int, road_grade: float, temperature_c: float,
                                                is_free_flow: bool = False):
        """Calculate single scenario fuel consumption using MOVES model."""
        vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
        if not vehicle_params:
            return 0.0

        base_rate = vehicle_params.get('fuel_efficiency_free_flow' if is_free_flow else 'fuel_efficiency_congested',
                                       0.1)
        speed_factor = max(0.7, 1.0 + self.MOVES_SPEED_CORRECTION_FACTOR * (max(5.0, speed) - 30.0) / 30.0)
        stop_factor = max(0.8, 1.0 + 0.1 * stops_per_km)
        grade_factor = max(0.9, 1.0 + 0.05 * road_grade)
        temp_factor = max(0.8, 1.0 + 0.02 * (max(-10.0, temperature_c) - 25.0) / 25.0)

        total_consumption = base_rate * distance * speed_factor * stop_factor * grade_factor * temp_factor * count
        return max(0.0, total_consumption)

    def calculate_barth_polynomial_emissions(self, speed_kmh: float, vehicle_type: str) -> float:
        """Calculate CO₂ emissions using actual Barth polynomial with proper scaling"""
        try:
            # Convert km/h to mph (Barth polynomial uses mph)
            speed_mph = speed_kmh / 1.609344

            # Barth coefficients
            b0, b1, b2, b3, b4 = 7.613549, -0.38564675, 0.01008333, -0.000116673, 0.000000527736

            # Calculate g CO₂/mile using the polynomial
            log_emissions = b0 + b1 * speed_mph + b2 * speed_mph ** 2 + b3 * speed_mph ** 3 + b4 * speed_mph ** 4
            emissions_g_per_mile = math.exp(log_emissions)

            # Convert to g CO₂/km and apply realistic scaling
            emissions_g_per_km = (emissions_g_per_mile / 1.609344) / 10.0  # Divided by 10 to get realistic values

            # Vehicle class adjustments
            class_adjustments = {
                'Motorcycles': 0.3,  # 30% of car emissions
                'Cars/SUVs': 1.0,  # Baseline
                'Coasters/Buses': 2.5,  # 2.5x car emissions
                'Trucks': 3.0,  # 3x car emissions
                'Tankers/Trailers': 4.0  # 4x car emissions
            }
            adjustment = class_adjustments.get(vehicle_type, 1.0)

            return emissions_g_per_km * adjustment

        except Exception as e:
            logger.error(f"Error in Barth polynomial calculation: {str(e)}")
            # Fallback realistic values (g CO₂/km)
            fallback_rates = {
                'Motorcycles': 50.0,
                'Cars/SUVs': 150.0,
                'Coasters/Buses': 400.0,
                'Trucks': 500.0,
                'Tankers/Trailers': 600.0
            }
            return fallback_rates.get(vehicle_type, 200.0)

    def validate_fuel_consumption(self, fuel_liters: float, vehicle_type: str, distance: float, count: int) -> float:
        """Validate and cap unrealistic fuel consumption values"""
        # Realistic maximum fuel consumption bounds (liters/km/vehicle)
        max_rates = {
            'Motorcycles': 0.05,  # 50 ml/km max
            'Cars/SUVs': 0.15,  # 150 ml/km max
            'Coasters/Buses': 0.4,  # 400 ml/km max
            'Trucks': 0.5,  # 500 ml/km max
            'Tankers/Trailers': 0.6  # 600 ml/km max
        }

        max_rate = max_rates.get(vehicle_type, 0.2)
        max_reasonable = max_rate * distance * count * 2.0  # Allow 2x buffer

        if fuel_liters > max_reasonable:
            logger.warning(
                f"Unrealistic fuel consumption {fuel_liters:.1f}L for {vehicle_type}, capping to {max_reasonable:.1f}L")
            return max_reasonable

        return fuel_liters

    def calculate_vkt_per_class(self, flow_rate: float, class_share: float,
                                segment_length: float, analysis_period: float) -> float:
        """Calculate Vehicle-Km Traveled per vehicle class"""
        return (class_share / 100.0) * flow_rate * segment_length * analysis_period

    def calculate_excess_co2_emissions(self, row, model: EmissionModelType):
        """Calculate EXCESS CO₂ emissions due to congestion (congested - free-flow)"""
        try:
            vehicle_type = row['Vehicle_Type']
            distance_km = float(row['Distance_KM'])

            free_flow_speed = float(row.get('Free_Flow_Speed_KPH', 60.0))
            congested_speed = float(row.get('Congested_Speed_KPH', 8.0))

            # FIXED: Use raw vehicle count without analysis period multiplication
            vehicle_count = float(row['Real_Vehicle_Count'])

            if model == EmissionModelType.BARTH:
                free_flow_emissions_g_km = self.calculate_barth_polynomial_emissions(free_flow_speed, vehicle_type)
                congested_emissions_g_km = self.calculate_barth_polynomial_emissions(congested_speed, vehicle_type)

                free_flow_total_kg = (free_flow_emissions_g_km * distance_km * vehicle_count) / 1000.0
                congested_total_kg = (congested_emissions_g_km * distance_km * vehicle_count) / 1000.0

                excess_emissions = max(0, congested_total_kg - free_flow_total_kg)

            elif model == EmissionModelType.MOVES:
                free_flow_fuel = self.calculate_moves_fuel_consumption_single(
                    speed=free_flow_speed, stops_per_km=0.5, vehicle_type=vehicle_type,
                    distance=distance_km, count=vehicle_count,
                    road_grade=float(row.get('Road_Grade', 0.0)), temperature_c=float(row.get('Temperature_C', 25.0)),
                    is_free_flow=True
                )
                congested_fuel = self.calculate_moves_fuel_consumption_single(
                    speed=congested_speed, stops_per_km=float(row.get('Stops_Per_KM', 2.0)),
                    vehicle_type=vehicle_type, distance=distance_km,
                    count=vehicle_count,
                    road_grade=float(row.get('Road_Grade', 0.0)), temperature_c=float(row.get('Temperature_C', 25.0)),
                    is_free_flow=False
                )

                fuel_type = self.vehicle_parameters.get(vehicle_type, {}).get('fuel_type', 'petrol')
                emission_factor = float(
                    row.get('Emission_Factor_Petrol', self.EMISSION_FACTOR_PETROL)) if fuel_type == 'petrol' else float(
                    row.get('Emission_Factor_Diesel', self.EMISSION_FACTOR_DIESEL))

                excess_emissions = max(0, (congested_fuel - free_flow_fuel) * emission_factor)

            else:  # BASIC model
                vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
                fuel_efficiency_free_flow = vehicle_params.get('fuel_efficiency_free_flow', 0.1)
                fuel_efficiency_congested = vehicle_params.get('fuel_efficiency_congested', 0.15)

                free_flow_fuel = distance_km * vehicle_count * fuel_efficiency_free_flow
                congested_fuel = distance_km * vehicle_count * fuel_efficiency_congested

                fuel_type = vehicle_params.get('fuel_type', 'petrol')
                emission_factor = float(
                    row.get('Emission_Factor_Petrol', self.EMISSION_FACTOR_PETROL)) if fuel_type == 'petrol' else float(
                    row.get('Emission_Factor_Diesel', self.EMISSION_FACTOR_DIESEL))

                excess_emissions = max(0, (congested_fuel - free_flow_fuel) * emission_factor)

            return excess_emissions

        except Exception as e:
            logger.error(f"Error in excess CO₂ calculation: {str(e)}")
            return 0.0

    def calculate_co2_emissions(self, row, model: EmissionModelType):
        """Calculate CO₂ emissions for delay period only (not total analysis period)"""
        try:
            # FIXED: Use RAW vehicle count without analysis period multiplication
            vehicle_count = float(row['Real_Vehicle_Count'])
            distance_km = float(row['Distance_KM'])
            vehicle_type = row['Vehicle_Type']

            # Use congested conditions only (like PDF) - FIXED: Now includes analysis period multiplier
            if model == EmissionModelType.BARTH:
                total_fuel = self.calculate_barth_fuel_consumption_single(
                    speed=float(row.get('Congested_Speed_KPH', 8.0)),
                    vehicle_type=vehicle_type,
                    distance=distance_km,
                    count=vehicle_count,  # Use count without analysis period multiplication
                    road_grade=float(row.get('Road_Grade', 0.0)),
                    is_congested=True,
                    stops_per_km=float(row.get('Stops_Per_KM', 2.0)),
                    idle_time=float(row.get('Idle_Time_Percentage', 0.3)) * float(
                        row.get('Congested_Travel_Time_Minutes', 45.0))
                )
            elif model == EmissionModelType.MOVES:
                total_fuel = self.calculate_moves_fuel_consumption_single(
                    speed=float(row.get('Congested_Speed_KPH', 8.0)),
                    stops_per_km=float(row.get('Stops_Per_KM', 2.0)),
                    vehicle_type=vehicle_type,
                    distance=distance_km,
                    count=vehicle_count,  # Use count without analysis period multiplication
                    road_grade=float(row.get('Road_Grade', 0.0)),
                    temperature_c=float(row.get('Temperature_C', 25.0)),
                    is_free_flow=False
                )
            else:  # BASIC model
                vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
                fuel_efficiency_congested = vehicle_params.get('fuel_efficiency_congested', 0.15)
                total_fuel = vehicle_count * distance_km * fuel_efficiency_congested

            fuel_type = self.vehicle_parameters.get(vehicle_type, {}).get('fuel_type', 'petrol')
            emission_factor = float(
                row.get('Emission_Factor_Petrol', self.EMISSION_FACTOR_PETROL)) if fuel_type == 'petrol' else float(
                row.get('Emission_Factor_Diesel', self.EMISSION_FACTOR_DIESEL))

            total_emissions = total_fuel * emission_factor

            logger.debug(
                f"CO₂ for {vehicle_type}: {total_fuel:.1f}L fuel × {emission_factor:.2f} kg/L = {total_emissions:.1f} kg CO₂")

            return total_emissions

        except (ValueError, KeyError) as e:
            logger.error(f"Error in calculate_co2_emissions: {str(e)}")
            return 0.0

    def calculate_barth_emissions(self, vehicle_type: str, count: int, distance: float, speed: float,
                                  acceleration: float, idle_time: float, road_grade: float):
        """Calculate emissions using Barth's model."""
        try:
            vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
            if not vehicle_params:
                logger.warning(f"No parameters for vehicle type: {vehicle_type}")
                return 0.0

            emission_factor = float(self.EMISSION_FACTOR_PETROL) if vehicle_params.get(
                'fuel_type') == 'petrol' else float(self.EMISSION_FACTOR_DIESEL)

            alpha = self.BARTH_COEFFICIENTS['alpha']
            beta = self.BARTH_COEFFICIENTS['beta']
            gamma = self.BARTH_COEFFICIENTS['gamma']
            traffic_flow = self.BARTH_COEFFICIENTS['traffic_flow']
            acceleration_factor = self.BARTH_COEFFICIENTS['acceleration']

            base_fuel_consumption = (
                    alpha * vehicle_params.get('engine_displacement_cc', 1500) +
                    beta * vehicle_params.get('weight_kg', 1000) / 1000 +
                    gamma * abs(road_grade)
            )

            speed_effect = traffic_flow * max(5.0, min(speed, 120.0)) / 50.0
            acceleration_effect = acceleration_factor * max(0, acceleration)
            total_fuel_per_km = max(0.01, base_fuel_consumption + speed_effect + acceleration_effect)
            distance_fuel = total_fuel_per_km * distance

            idle_fuel = 0
            if idle_time > 0:
                idle_rate = self.BARTH_IDLE_FUEL_CONSUMPTION_PETROL if vehicle_params.get(
                    'fuel_type') == 'petrol' else self.BARTH_IDLE_FUEL_CONSUMPTION_DIESEL
                idle_fuel = idle_rate * idle_time

            total_fuel = (distance_fuel + idle_fuel) * count
            total_emissions = total_fuel * emission_factor
            return max(0, total_emissions)

        except (ValueError, KeyError, IndexError) as e:
            logger.error(f"Error in calculate_barth_emissions: {str(e)}")
            return 0.0

    def calculate_moves_emissions(self, vehicle_type: str, count: int, distance: float, speed: float,
                                  acceleration: float, stops_per_km: float, road_grade: float, temperature_c: float):
        """Calculate emissions using MOVES model."""
        try:
            vehicle_params = self.vehicle_parameters.get(vehicle_type, {})
            if not vehicle_params:
                logger.warning(f"No parameters for vehicle type: {vehicle_type}")
                return 0.0

            base_rate = self.MOVES_BASE_EMISSION_RATE_PETROL / 1000.0 if vehicle_params.get(
                'fuel_type') == 'petrol' else self.MOVES_BASE_EMISSION_RATE_DIESEL / 1000.0

            speed_correction = max(0.5, 1.0 + self.MOVES_SPEED_CORRECTION_FACTOR * (max(5.0, speed) - 30.0) / 30.0)
            acceleration_correction = max(0.8, 1.0 + self.MOVES_ACCEL_CORRECTION_FACTOR * max(0, acceleration))
            stop_correction = max(0.9, 1.0 + 0.05 * stops_per_km)
            grade_correction = max(0.9, 1.0 + 0.02 * abs(road_grade))
            temp_correction = max(0.8, 1.0 + 0.01 * (max(-10.0, temperature_c) - 25.0) / 25.0)

            emissions_per_vehicle = base_rate * speed_correction * acceleration_correction * stop_correction * grade_correction * temp_correction * distance
            total_emissions = emissions_per_vehicle * count
            return max(0, total_emissions)

        except (ValueError, KeyError) as e:
            logger.error(f"Error in calculate_moves_emissions: {str(e)}")
            return 0.0

    def _get_metric_value_for_pdf(self, row, metric_name, metric_unit, metric):
        """Helper method to safely get metric value for PDF generation."""
        try:
            possible_columns = [
                f'{metric_name} ({metric_unit})',
                metric_name,
                metric_name.split(' (')[0] if '(' in metric_name else metric_name,
                'Excess Fuel Used (L)' if 'Excess Fuel' in metric_name else metric_name,
                'Excess_Fuel_Used' if 'Excess Fuel' in metric_name else metric_name,
                'Productivity Loss (₦)' if metric == CalculationMetric.PRODUCTIVITY_LOSS else metric_name,
                'CO₂ Emissions (kg)' if metric == CalculationMetric.CO2_EMISSIONS else metric_name,
                'Excess_Fuel' if 'Excess Fuel' in metric_name else metric_name,
                'Productivity_Loss' if metric == CalculationMetric.PRODUCTIVITY_LOSS else metric_name,
                'CO2_Emissions' if metric == CalculationMetric.CO2_EMISSIONS else metric_name
            ]

            for col in possible_columns:
                if col in row:
                    value = row[col]
                    if pd.isna(value):
                        continue
                    if metric == CalculationMetric.EXCESS_FUEL:
                        return f"{float(value):,.1f} L"
                    elif metric == CalculationMetric.CO2_EMISSIONS:
                        return f"{float(value):,.1f} kg"
                    else:
                        return f"₦{float(value):,.0f}"

            for col_name in row.keys():
                if metric_name.lower().replace(' ', '_') in col_name.lower().replace(' ', '_'):
                    value = row[col_name]
                    if pd.isna(value):
                        continue
                    if metric == CalculationMetric.EXCESS_FUEL:
                        return f"{float(value):,.1f} L"
                    elif metric == CalculationMetric.CO2_EMISSIONS:
                        return f"{float(value):,.1f} kg"
                    else:
                        return f"₦{float(value):,.0f}"

            logger.warning(
                f"Could not find metric column for {metric_name} in row. Available columns: {list(row.keys())}")
            if metric == CalculationMetric.EXCESS_FUEL:
                return "0.0 L"
            elif metric == CalculationMetric.CO2_EMISSIONS:
                return "0.0 kg"
            else:
                return "₦0"

        except Exception as e:
            logger.error(f"Error getting metric value for PDF: {str(e)}")
            if metric == CalculationMetric.EXCESS_FUEL:
                return "0.0 L"
            elif metric == CalculationMetric.CO2_EMISSIONS:
                return "0.0 kg"
            else:
                return "₦0"

    def generate_report(self, metric: CalculationMetric, model: EmissionModelType = None):
        """Generate a report DataFrame for the selected metric with FIXED occupancy rate handling."""
        if model is None:
            model = self.emission_model

        try:
            data = []
            for state, cities in self.state_city_data.items():
                for city, df in cities.items():
                    for _, row in df.iterrows():
                        # Get the REAL occupancy rate from user input
                        occupancy = float(row.get('Real_VOR', self.vehicle_parameters.get(row['Vehicle_Type'], {}).get(
                            'occupancy_avg', 1.0)))

                        record = {
                            'State': state,
                            'City': city,
                            'Vehicle_Type': row['Vehicle_Type'],
                            'Vehicle Count': float(row['Real_Vehicle_Count']),  # FIXED: No analysis period multiplication
                            'Occupancy': occupancy
                        }

                        # Create a modified row with the correct occupancy for calculations
                        modified_row = row.copy()
                        modified_row['Real_VOR'] = occupancy

                        if metric == CalculationMetric.PRODUCTIVITY_LOSS:
                            record['Productivity Loss (₦)'] = self.calculate_productivity_loss(modified_row)
                        elif metric == CalculationMetric.EXCESS_FUEL:
                            excess_fuel_cost = self.calculate_excess_fuel(modified_row, model)
                            record['Excess Fuel Cost (₦)'] = excess_fuel_cost
                        elif metric == CalculationMetric.CO2_EMISSIONS:
                            record['CO₂ Emissions (kg)'] = self.calculate_co2_emissions(modified_row, model)

                        data.append(record)
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def get_vehicle_distribution(self, state: str, city: str):
        """Get vehicle distribution for a specific city within a state."""
        try:
            if state not in self.state_city_data or city not in self.state_city_data[state]:
                logger.warning(f"No data for city: {city}, state: {state}")
                return {}
            df = self.state_city_data[state][city]
            distribution = {vehicle_type: 0 for vehicle_type in valid_vehicle_types}
            for _, row in df.iterrows():
                distribution[row['Vehicle_Type']] += float(row['Real_Vehicle_Count'])  # FIXED: No analysis period multiplication
            return distribution
        except Exception as e:
            logger.error(f"Error getting vehicle distribution for {city}, {state}: {str(e)}", exc_info=True)
            return {}

    def generate_chart_images(self, metric: CalculationMetric):
        """Generate chart images for the report."""
        try:
            chart_images = {}
            temp_dir = Path('temp_charts')
            temp_dir.mkdir(exist_ok=True)

            for existing_chart in temp_dir.glob("*.png"):
                try:
                    existing_chart.unlink()
                except:
                    pass

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            for state in self.state_city_data:
                for city in self.state_city_data[state]:
                    distribution = self.get_vehicle_distribution(state, city)
                    labels = list(distribution.keys())
                    sizes = list(distribution.values())
                    if sum(sizes) == 0:
                        continue
                    plt.figure(figsize=(8, 6))
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)], startangle=90,
                            textprops={'fontsize': 12})
                    plt.title(f'Vehicle Distribution in {city}, {state}', fontsize=14, pad=15, weight='bold')
                    chart_path = temp_dir / f'vehicle_distribution_{state}_{city.replace(" ", "_").lower()}.png'
                    plt.savefig(chart_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    chart_images[f'vehicle_distribution_{state}_{city}'] = str(chart_path)

                    df = self.state_city_data[state][city]
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Vehicle_Type', y='Real_Vehicle_Count', data=df,
                                palette=colors[:len(df['Vehicle_Type'].unique())])
                    plt.title(f'Vehicle Counts in {city}, {state}', fontsize=14, pad=15, weight='bold')
                    plt.xticks(rotation=45, ha='right')
                    chart_path = temp_dir / f'vehicle_counts_{state}_{city.replace(" ", "_").lower()}.png'
                    plt.savefig(chart_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    chart_images[f'vehicle_counts_{state}_{city}'] = str(chart_path)

            plt.figure(figsize=(10, 6))
            totals = {f"{state}_{city}": sum(self.get_vehicle_distribution(state, city).values())
                      for state in self.state_city_data
                      for city in self.state_city_data[state]}
            sns.barplot(x=list(totals.keys()), y=list(totals.values()), palette=colors[:len(totals)])
            plt.title('Total Vehicles Across Cities', fontsize=14, pad=15, weight='bold')
            plt.xticks(rotation=45, ha='right')
            chart_path = temp_dir / 'total_vehicles.png'
            plt.savefig(chart_path, bbox_inches='tight', dpi=150)
            plt.close()
            chart_images['total_vehicles'] = str(chart_path)

            results = self.calculate_metric(metric=metric)
            metric_key = f'total_{metric.value}'
            metric_data = [results['city_results'][state][city][metric_key]
                           for state in results['city_results']
                           for city in results['city_results'][state]]
            city_keys = [f"{state}_{city}"
                         for state in results['city_results']
                         for city in results['city_results'][state]]
            if any(x > 0 for x in metric_data):
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(metric_data)), metric_data, color=colors[:len(metric_data)])
                plt.title(f'{metric.name.replace("_", " ").title()} Across Cities', fontsize=14, pad=15, weight='bold')

                if metric == CalculationMetric.PRODUCTIVITY_LOSS:
                    plt.ylabel('Productivity Loss (₦)', fontsize=12)
                elif metric == CalculationMetric.EXCESS_FUEL:
                    plt.ylabel('Excess Fuel Cost (₦)', fontsize=12)
                else:
                    plt.ylabel('CO₂ Emissions (kg)', fontsize=12)

                plt.xticks(range(len(metric_data)), city_keys, rotation=45, ha='right')
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height):,}', ha='center',
                                 va='bottom', fontsize=10)
                plt.tight_layout()
                chart_path = temp_dir / f'{metric.value}.png'
                plt.savefig(chart_path, bbox_inches='tight', dpi=150)
                plt.close()
                chart_images[metric.value] = str(chart_path)

            return chart_images
        except Exception as e:
            logger.error(f"Error generating chart images: {str(e)}", exc_info=True)
            return {}

    def generate_pdf_report(self, output_path: str, metric: CalculationMetric, model: EmissionModelType = None):
        """Generate a PDF report for the selected metric with proper chart handling."""
        if model is None:
            model = self.emission_model

        if not self.validate_model_selection(metric, model):
            logger.warning(f"Using model {model.value} for {metric.value} despite validation warning")

        try:
            if not output_path.endswith('.pdf'):
                output_path += '.pdf'
            output_path = Path(output_path)
            os.makedirs(output_path.parent, exist_ok=True)

            results = self.calculate_metric(metric=metric, model=model)
            if not results['city_results']:
                logger.error("No analysis results available for PDF generation")
                return None

            report_df = self.generate_report(metric=metric, model=model)
            if report_df.empty:
                logger.error("No report data available for PDF generation")
                return None

            logger.info("Generating chart images for PDF report...")
            chart_images = self.generate_chart_images(metric=metric)
            logger.info(f"Generated {len(chart_images)} chart images")

            import time
            time.sleep(2)

            available_charts = {}
            temp_charts_dir = Path('temp_charts')

            if temp_charts_dir.exists():
                for chart_file in temp_charts_dir.glob('*.png'):
                    if chart_file.exists() and chart_file.stat().st_size > 0:
                        chart_name = chart_file.stem
                        available_charts[chart_name] = str(chart_file.absolute())
                        logger.info(f"Chart file verified: {chart_file} ({chart_file.stat().st_size} bytes)")
                    else:
                        logger.warning(f"Chart file not found or empty: {chart_file}")

            if not available_charts and chart_images:
                for chart_name, chart_path in chart_images.items():
                    chart_file = Path(chart_path)
                    if chart_file.exists() and chart_file.stat().st_size > 0:
                        available_charts[chart_name] = str(chart_file.absolute())
                        logger.info(f"Using chart from dict: {chart_file}")

            logger.info(f"Available charts for PDF: {list(available_charts.keys())}")

            # FIXED: Correct metric names and units
            if metric == CalculationMetric.PRODUCTIVITY_LOSS:
                metric_name = "Productivity Loss"
                metric_unit = "₦"
                display_unit = "naira"
            elif metric == CalculationMetric.EXCESS_FUEL:
                metric_name = "Excess Fuel Cost"
                metric_unit = "₦"
                display_unit = "naira"
            else:
                metric_name = "CO₂ Emissions"
                metric_unit = "kg"
                display_unit = "kg"

            fuel_price_petrol = self.DEFAULT_FUEL_PRICE_PETROL
            fuel_price_diesel = self.DEFAULT_FUEL_PRICE_DIESEL

            if not self.data.empty:
                petrol_prices = self.data['Fuel_Cost_Per_Liter_Petrol'].dropna()
                diesel_prices = self.data['Fuel_Cost_Per_Liter_Diesel'].dropna()

                if not petrol_prices.empty:
                    fuel_price_petrol = float(petrol_prices.iloc[0])
                    logger.info(f"Extracted petrol price from data: ₦{fuel_price_petrol:,.2f}/L")

                if not diesel_prices.empty:
                    fuel_price_diesel = float(diesel_prices.iloc[0])
                    logger.info(f"Extracted diesel price from data: ₦{fuel_price_diesel:,.2f}/L")

            fuel_price_petrol_str = f"₦{fuel_price_petrol:,.2f}/L"
            fuel_price_diesel_str = f"₦{fuel_price_diesel:,.2f}/L"

            free_flow_time = float(self.data['Free_Flow_Time_Minutes'].dropna().iloc[0] if not self.data[
                'Free_Flow_Time_Minutes'].dropna().empty else 4.0)
            congested_time = float(self.data['Congested_Travel_Time_Minutes'].dropna().iloc[0] if not self.data[
                'Congested_Travel_Time_Minutes'].dropna().empty else 45.0)
            corridor_length = float(self.data['Distance_KM'].dropna().iloc[0] if not self.data[
                'Distance_KM'].dropna().empty else self.CORRIDOR_LENGTH)
            value_per_minute = float(self.data['Value_Per_Minute_Naira'].dropna().iloc[0] if not self.data[
                'Value_Per_Minute_Naira'].dropna().empty else self.VALUE_PER_MINUTE)
            free_flow_speed = float(self.data['Free_Flow_Speed_KPH'].dropna().iloc[0] if not self.data[
                'Free_Flow_Speed_KPH'].dropna().empty else 60.0)
            congested_speed = float(self.data['Congested_Speed_KPH'].dropna().iloc[0] if not self.data[
                'Congested_Speed_KPH'].dropna().empty else 8.0)
            acceleration = float(self.data['Avg_Acceleration'].dropna().iloc[0] if not self.data[
                'Avg_Acceleration'].dropna().empty else 0.5)
            idle_time_pct = float(self.data['Idle_Time_Percentage'].dropna().iloc[0] if not self.data[
                'Idle_Time_Percentage'].dropna().empty else 0.3)
            stops_per_km = float(
                self.data['Stops_Per_KM'].dropna().iloc[0] if not self.data['Stops_Per_KM'].dropna().empty else 2.0)
            road_grade = float(
                self.data['Road_Grade'].dropna().iloc[0] if not self.data['Road_Grade'].dropna().empty else 0.0)
            temperature = float(
                self.data['Temperature_C'].dropna().iloc[0] if not self.data['Temperature_C'].dropna().empty else 25.0)

            max_city = max(
                ((state, city) for state in results['city_results'] for city in results['city_results'][state]),
                key=lambda sc: results['city_results'][sc[0]][sc[1]]['total_vehicles'],
                default=None
            )
            highest_city = f"{max_city[1]}, {max_city[0]}" if max_city else "Unknown"

            # FIXED: Correct display values for all metrics
            if metric == CalculationMetric.EXCESS_FUEL:
                display_value = f"₦{results['total_summary'][f'total_{metric.value}_all_cities']:,.0f}"
            elif metric == CalculationMetric.CO2_EMISSIONS:
                display_value = f"{results['total_summary'][f'total_{metric.value}_all_cities']:,.0f} kg"
            else:  # PRODUCTIVITY_LOSS
                display_value = f"₦{results['total_summary'][f'total_{metric.value}_all_cities']:,.0f}"

            homepage_data = {
                'title': "Nigeria Traffic Analysis System",
                'subtitle': f"{metric_name} Analysis Report",
                'stats': [
                    {'label': 'Total Vehicles Analyzed',
                     'value': f"{results['total_summary']['total_vehicles_all_cities']:,}"},
                    {'label': 'Total People Affected',
                     'value': f"{results['total_summary']['total_people_all_cities']:,}"},
                    {'label': f'Total {metric_name}',
                     'value': display_value}  # FIXED: Use corrected display value
                ],
                'features': [
                    "Accurate vehicle counting and classification",
                    "Real-time congestion analysis",
                    f"Emission modeling ({model.value.title()})",
                    f"{metric_name} assessment"
                ],
                'how_it_works': [
                    "Collect traffic data via sensors or manual input",
                    f"Analyze vehicle counts and congestion for {metric_name.lower()}",
                    f"Calculate {metric_name.lower()} using {model.value.title()} formula",
                    "Generate comprehensive reports and visualizations"
                ],
                'methodology': {
                    'description': f"This assessment uses vehicle-specific parameters to calculate {metric_name.lower()}:",
                    'parameters': [
                        f"Fuel Prices: {fuel_price_petrol_str} (petrol), {fuel_price_diesel_str} (diesel)",
                        f"Emission Factors: {self.EMISSION_FACTOR_PETROL:,.2f} kg CO₂/L (petrol), {self.EMISSION_FACTOR_DIESEL:,.2f} kg CO₂/L (diesel)",
                        f"Productivity Value: ₦{value_per_minute:,.2f}/minute",
                        f"Corridor Length: {corridor_length:,.1f} km",
                        f"Time Parameters: Free flow: {free_flow_time:.1f} min, Congested: {congested_time:.1f} min",
                        f"Advanced Parameters: Free flow speed: {free_flow_speed:.1f} km/h, Congested speed: {congested_speed:.1f} km/h, Acceleration: {acceleration:.1f} m/s², Idle time: {idle_time_pct:.1%}, Stops per km: {stops_per_km:.1f}, Road grade: {road_grade:.1f}%, Temperature: {temperature:.1f}°C"
                    ] if model != EmissionModelType.BASIC else [
                        f"Fuel Prices: {fuel_price_petrol_str} (petrol), {fuel_price_diesel_str} (diesel)",
                        f"Emission Factors: {self.EMISSION_FACTOR_PETROL:,.2f} kg CO₂/L (petrol), {self.EMISSION_FACTOR_DIESEL:,.2f} kg CO₂/L (diesel)",
                        f"Productivity Value: ₦{value_per_minute:,.2f}/minute",
                        f"Corridor Length: {corridor_length:,.1f} km",
                        f"Time Parameters: Free flow: {free_flow_time:.1f} min, Congested: {congested_time:.1f} min"
                    ]
                },
                'recommendations': {
                    'key_findings': [
                        f"{highest_city} shows the highest congestion impact, accounting for approximately {(results['city_results'][max_city[0]][max_city[1]]['total_vehicles'] / results['total_summary']['total_vehicles_all_cities'] * 100 if results['total_summary']['total_vehicles_all_cities'] > 0 else 0):,.1f}% of total vehicles." if max_city else "No data available.",
                        f"Evening rush hour congestion affects nearly {results['total_summary']['total_people_all_cities']:,} commuters across all cities.",
                        f"Total {metric_name.lower()} is {display_value}.",
                        f"Analysis performed using {model.value.upper()} emission model."
                    ],
                    'strategic_recommendations': [
                        f"Immediate Interventions: Implement targeted traffic management systems in {highest_city}." if max_city else "Immediate Interventions: Implement targeted traffic management systems.",
                        "Public Transportation: Enhance options to reduce private vehicle use during peak hours.",
                        f"Policy Measures: Consider congestion pricing to reduce {metric_name.lower()}.",
                        "Data Collection: Expand monitoring to understand daily and weekly patterns."
                    ]
                }
            }

            doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm,
                                    topMargin=2 * cm, bottomMargin=2 * cm)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='TitleStyle', fontSize=18, alignment=TA_CENTER, spaceAfter=12,
                                      textColor=colors.HexColor('#1f77b4')))
            styles.add(ParagraphStyle(name='SubtitleStyle', fontSize=14, alignment=TA_CENTER, spaceAfter=12))
            styles.add(ParagraphStyle(name='SectionTitle', fontSize=12, spaceBefore=12, spaceAfter=6,
                                      fontName='Helvetica-Bold'))
            styles.add(ParagraphStyle(name='Footer', fontSize=8, alignment=TA_CENTER, textColor=colors.grey))
            styles.add(ParagraphStyle(name='ModelBadge', fontSize=10, fontName='Helvetica-Bold', textColor=colors.white,
                                      backColor=colors.HexColor('#1f77b4'), spaceAfter=6, leading=12))

            story = []
            story.append(Paragraph(homepage_data['title'], styles['TitleStyle']))
            story.append(Paragraph(homepage_data['subtitle'], styles['SubtitleStyle']))
            story.append(Paragraph(
                f"Generated on {self.analysis_date} at {self.analysis_time} | Emission Model: {model.value.capitalize()}",
                styles['Normal']))
            story.append(Spacer(1, 0.5 * cm))

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
            story.append(stats_table)
            story.append(Spacer(1, 0.5 * cm))

            story.append(Paragraph("Features", styles['SectionTitle']))
            for feature in homepage_data['features']:
                story.append(Paragraph(f"• {feature}", styles['Normal']))
            story.append(Spacer(1, 0.5 * cm))

            story.append(Paragraph("How It Works", styles['SectionTitle']))
            for step in homepage_data['how_it_works']:
                story.append(Paragraph(f"• {step}", styles['Normal']))
            story.append(Spacer(1, 0.5 * cm))

            story.append(Paragraph("Methodology", styles['SectionTitle']))
            story.append(Paragraph(homepage_data['methodology']['description'], styles['Normal']))
            for param in homepage_data['methodology']['parameters']:
                story.append(Paragraph(f"• {param}", styles['Normal']))
            story.append(Spacer(1, 0.5 * cm))

            story.append(PageBreak())
            story.append(Paragraph(f"{metric_name} Analysis Report", styles['TitleStyle']))
            story.append(Paragraph(
                f"Analysis for {len([city for state in self.state_city_data for city in self.state_city_data[state]])} Nigerian Cities",
                styles['Normal']))
            story.append(Paragraph(f"Generated on: {self.analysis_date} at {self.analysis_time}", styles['Normal']))
            story.append(Paragraph("Analysis Period: Evening Rush Hour (18:00)", styles['Normal']))
            story.append(Spacer(1, 0.5 * cm))

            model_description = {
                EmissionModelType.BASIC: "Using simple fuel consumption-based emission calculations with fixed emission factors.",
                EmissionModelType.BARTH: "Using Barth's comprehensive fuel consumption model with acceleration, deceleration, and idle time factors.",
                EmissionModelType.MOVES: "Using EPA MOVES-like emission model with speed, acceleration, and vehicle standard corrections."
            }.get(model, f"Using {model.value.upper()} emission calculation model.")
            story.append(Paragraph(f"<b>{model.value.upper()}</b> {model_description}", styles['ModelBadge']))
            story.append(Spacer(1, 0.3 * cm))

            story.append(Paragraph("Executive Summary", styles['SectionTitle']))
            story.append(Paragraph(
                f"This report provides a comprehensive analysis of traffic congestion impacts across Nigerian cities, focusing on {metric_name.lower()}.",
                styles['Normal']))
            metric_key = f"total_{metric.value}_all_cities"

            # FIXED: Correct summary table units
            if metric == CalculationMetric.EXCESS_FUEL:
                summary_unit = "naira"
                summary_value = f"₦{results['total_summary'][metric_key]:,.0f}"
            elif metric == CalculationMetric.CO2_EMISSIONS:
                summary_unit = "kg"
                summary_value = f"{results['total_summary'][metric_key]:,.0f} kg"
            else:
                summary_unit = "naira"
                summary_value = f"₦{results['total_summary'][metric_key]:,.0f}"

            summary_data = [
                ["Total Vehicles", f"{results['total_summary']['total_vehicles_all_cities']:,}", "vehicles"],
                ["People Affected", f"{results['total_summary']['total_people_all_cities']:,}", "commuters"],
                [f"Total {metric_name}", summary_value, summary_unit]
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
            story.append(summary_table)
            story.append(Spacer(1, 0.5 * cm))

            story.append(PageBreak())
            story.append(Paragraph("Grand Totals Across All Cities", styles['SectionTitle']))

            # FIXED: Correct totals table units
            if metric == CalculationMetric.EXCESS_FUEL:
                totals_unit = "naira"
                totals_value = f"₦{results['total_summary'][metric_key]:,.0f}"
            elif metric == CalculationMetric.CO2_EMISSIONS:
                totals_unit = "kg"
                totals_value = f"{results['total_summary'][metric_key]:,.0f} kg"
            else:
                totals_unit = "naira"
                totals_value = f"₦{results['total_summary'][metric_key]:,.0f}"

            totals = [
                ["Performance Metric", "Value", "Unit"],
                ["Total Vehicles Observed", f"{results['total_summary']['total_vehicles_all_cities']:,}", "vehicles"],
                ["Total People Affected", f"{results['total_summary']['total_people_all_cities']:,}", "people"],
                [f"Total {metric_name}", totals_value, totals_unit]
            ]
            totals_table = Table(totals, colWidths=[8 * cm, 6 * cm, 2 * cm])
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
            story.append(totals_table)
            story.append(Spacer(1, 0.5 * cm))

            story.append(PageBreak())
            story.append(Paragraph("Per-City Analysis", styles['SectionTitle']))
            for state in results['city_results']:
                for city in results['city_results'][state]:
                    city_data = results['city_results'][state][city]

                    # FIXED: Correct city table display
                    if metric == CalculationMetric.EXCESS_FUEL:
                        city_value = f"₦{city_data[f'total_{metric.value}']:,.0f}"
                    elif metric == CalculationMetric.CO2_EMISSIONS:
                        city_value = f"{city_data[f'total_{metric.value}']:,.0f} kg"
                    else:
                        city_value = f"₦{city_data[f'total_{metric.value}']:,.0f}"

                    city_table_data = [
                        ["Metric", "Value"],
                        ["Vehicles", f"{city_data['total_vehicles']:,}"],
                        ["People Affected", f"{city_data['total_people']:,}"],
                        [f"{metric_name}", city_value]
                    ]
                    city_table = Table(city_table_data, colWidths=[8 * cm, 8 * cm])
                    city_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
                        ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ]))
                    story.append(Paragraph(f"{city}, {state}", styles['SectionTitle']))
                    story.append(city_table)
                    story.append(Spacer(1, 0.3 * cm))

            story.append(PageBreak())
            story.append(Paragraph("Visual Analysis", styles['SectionTitle']))

            charts_added = False
            for chart_key, chart_path in available_charts.items():
                if 'vehicle_distribution' in chart_key:
                    try:
                        parts = chart_key.replace('vehicle_distribution_', '').split('_')
                        if len(parts) >= 2:
                            state = parts[0].replace('_', ' ').title()
                            city = parts[1].replace('_', ' ').title()
                            story.append(Paragraph(f"{city}, {state} Vehicle Distribution", styles['SectionTitle']))
                            story.append(Image(chart_path, width=12 * cm, height=10 * cm))
                            story.append(Spacer(1, 0.3 * cm))
                            charts_added = True
                    except Exception as e:
                        logger.error(f"Failed to add distribution chart {chart_key}: {str(e)}")

            for chart_key, chart_path in available_charts.items():
                if 'vehicle_counts' in chart_key:
                    try:
                        parts = chart_key.replace('vehicle_counts_', '').split('_')
                        if len(parts) >= 2:
                            state = parts[0].replace('_', ' ').title()
                            city = parts[1].replace('_', ' ').title()
                            story.append(Paragraph(f"{city}, {state} Vehicle Counts", styles['SectionTitle']))
                            story.append(Image(chart_path, width=12 * cm, height=10 * cm))
                            story.append(Spacer(1, 0.3 * cm))
                            charts_added = True
                    except Exception as e:
                        logger.error(f"Failed to add count chart {chart_key}: {str(e)}")

            summary_charts = {
                'total_vehicles': 'Total Vehicles Across Cities',
                metric.value: f'{metric_name} Across Cities'
            }

            for chart_key, chart_title in summary_charts.items():
                if chart_key in available_charts:
                    chart_path = available_charts[chart_key]
                    try:
                        story.append(Paragraph(chart_title, styles['SectionTitle']))
                        story.append(Image(chart_path, width=14 * cm, height=10 * cm))
                        story.append(Spacer(1, 0.3 * cm))
                        charts_added = True
                    except Exception as e:
                        logger.error(f"Failed to add summary chart {chart_key}: {str(e)}")
                else:
                    for available_key, chart_path in available_charts.items():
                        if chart_key in available_key:
                            try:
                                story.append(Paragraph(chart_title, styles['SectionTitle']))
                                story.append(Image(chart_path, width=14 * cm, height=10 * cm))
                                story.append(Spacer(1, 0.3 * cm))
                                charts_added = True
                                break
                            except Exception as e:
                                logger.error(f"Failed to add matched chart {available_key}: {str(e)}")

            if not charts_added:
                story.append(Paragraph("No charts available for display", styles['Normal']))
                story.append(Spacer(1, 0.5 * cm))

            story.append(PageBreak())
            story.append(Paragraph("Detailed Report", styles['SectionTitle']))
            table_data = [['State', 'City', 'Vehicle_Type', 'Vehicle Count', 'Occupancy', f'{metric_name}']]
            for row in report_df.to_dict('records'):
                table_data.append([
                    row['State'],
                    row['City'],
                    row['Vehicle_Type'],
                    f"{row['Vehicle Count']:,}",
                    f"{row['Occupancy']:.1f}",
                    self._get_metric_value_for_pdf(row, metric_name, metric_unit, metric)
                ])
            results_table = Table(table_data, colWidths=[3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm, 4 * cm])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (1, -1), 'LEFT'),
                ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(results_table)
            story.append(Spacer(1, 0.5 * cm))

            story.append(PageBreak())
            story.append(Paragraph("Intermediate Calculations", styles['SectionTitle']))
            story.append(Paragraph("Detailed breakdown of calculations for verification:", styles['Normal']))
            story.append(Spacer(1, 0.3 * cm))

            intermediate_calcs_added = False
            for state in self.state_city_data:
                for city in self.state_city_data[state]:
                    df = self.state_city_data[state][city]
                    for _, row in df.iterrows():
                        calculations = self.get_intermediate_calculations(row, metric, model)

                        story.append(Paragraph(f"{row['Vehicle_Type']} in {city}, {state}", styles['SectionTitle']))

                        for section_name, section_data in calculations.items():
                            story.append(Paragraph(f"{section_name.replace('_', ' ').title()}:", styles['Normal']))

                        if isinstance(section_data, dict):
                            for key, value in section_data.items():
                                if isinstance(value, float):
                                    display_value = f"{value:,.2f}"
                                else:
                                    display_value = str(value)
                                story.append(Paragraph(f"  {key.replace('_', ' ').title()}: {display_value}",
                                                       styles['Normal']))
                        else:
                            story.append(Paragraph(f"  {section_data}", styles['Normal']))

                        story.append(Spacer(1, 0.2 * cm))
                        intermediate_calcs_added = True

            if not intermediate_calcs_added:
                story.append(Paragraph("No intermediate calculations available", styles['Normal']))

            story.append(PageBreak())
            story.append(Paragraph("Conclusions & Recommendations", styles['SectionTitle']))
            story.append(Paragraph("Key Findings", styles['SectionTitle']))
            for finding in homepage_data['recommendations']['key_findings']:
                story.append(Paragraph(f"• {finding}", styles['Normal']))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph("Strategic Recommendations", styles['SectionTitle']))
            for recommendation in homepage_data['recommendations']['strategic_recommendations']:
                story.append(Paragraph(f"• {recommendation}", styles['Normal']))
            story.append(Spacer(1, 0.5 * cm))

            story.append(Paragraph(
                f"Generated on {self.analysis_date} at {self.analysis_time} by Nigeria Traffic Analysis System<br/>&copy; {datetime.now().year} | Powered by Opygoal Technology Ltd | Developed by Oladotun Ajakaiye",
                styles['Footer']))

            doc.build(story)
            logger.info(
                f"PDF report generated at {output_path} with {charts_added} charts and intermediate calculations")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
            return None

    def get_available_models(self):
        """Return list of available emission models."""
        return [{"value": model.value, "name": model.display_name} for model in EmissionModelType]

    def get_available_metrics(self):
        """Return list of available calculation metrics."""
        return [metric.value for metric in CalculationMetric]

    def get_states(self):
        """Return list of states with data."""
        return list(self.state_city_data.keys())

    def get_cities_by_state(self, state_name):
        """Return list of cities for a given state."""
        if state_name in self.state_city_data:
            return list(self.state_city_data[state_name].keys())
        return []

    def get_state_totals(self):
        """Get vehicle totals by state."""
        state_totals = {}
        for state, cities in self.state_city_data.items():
            total = 0
            for city, df in cities.items():
                total += df['Real_Vehicle_Count'].sum()  # FIXED: No analysis period multiplication
            state_totals[state] = total
        return state_totals