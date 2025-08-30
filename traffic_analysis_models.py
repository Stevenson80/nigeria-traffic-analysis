import pandas as pd
import logging
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
from enum import Enum

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmissionModelType(Enum):
    BASIC = "basic"
    BARTH = "barth"
    MOVES = "moves"


class TrafficAnalysisModel:
    FUEL_PRICE_PETROL = 850.0
    FUEL_PRICE_DIESEL = 1000.0
    EMISSION_FACTOR_PETROL = 2.31
    EMISSION_FACTOR_DIESEL = 2.68
    VALUE_PER_MINUTE = 7.29
    CORRIDOR_LENGTH = 6.0
    BARTH_IDLE_FUEL_CONSUMPTION_PETROL = 0.6
    BARTH_IDLE_FUEL_CONSUMPTION_DIESEL = 0.8
    BARTH_ACCEL_FACTOR_PETROL = 0.00035
    BARTH_ACCEL_FACTOR_DIESEL = 0.00045
    MOVES_BASE_EMISSION_RATE_PETROL = 2.0
    MOVES_BASE_EMISSION_RATE_DIESEL = 2.5
    MOVES_SPEED_CORRECTION_FACTOR = 0.05
    MOVES_ACCEL_CORRECTION_FACTOR = 0.1

    # Barth coefficients for emission calculations
    BARTH_COEFFICIENTS = {
        'alpha': 0.0003,  # Base coefficient for engine displacement
        'beta': 0.0025,  # Coefficient for vehicle weight
        'gamma': 0.0018,  # Coefficient for road gradient
        'traffic_flow': 0.15,  # Traffic flow coefficient
        'road_gradient': 0.10,  # Road gradient coefficient
        'acceleration': 0.04  # Acceleration coefficient
    }

    # Vehicle-specific emission factors (kg CO₂/km)
    VEHICLE_EMISSION_FACTORS = {
        'Motorcycles': 0.12,
        'Cars': 0.18,
        'SUVs': 0.22,
        'Sedans': 0.19,
        'Wagons': 0.20,
        'Short Buses': 0.28,
        'Minibusses': 0.32,
        'Long Buses': 0.40,
        'Truck': 0.50,
        'Tanker and Trailer': 0.70
    }

    def __init__(self, csv_file_path=None, emission_model: EmissionModelType = EmissionModelType.BASIC):
        self.vehicle_parameters = {
            'Motorcycles': {'class': 1, 'occupancy_min': 1.1, 'occupancy_max': 1.7, 'occupancy_avg': 1.4,
                            'fuel_consumption_l_per_km_free_flow': 0.03, 'fuel_consumption_l_per_km_congested': 0.045,
                            'fuel_type': 'petrol', 'weight_kg': 150, 'engine_displacement_cc': 125, 'euro_standard': 3},
            'Cars': {'class': 2, 'occupancy_min': 4.9, 'occupancy_max': 5.5, 'occupancy_avg': 5.2,
                     'fuel_consumption_l_per_km_free_flow': 0.1, 'fuel_consumption_l_per_km_congested': 0.15,
                     'fuel_type': 'petrol', 'weight_kg': 1200, 'engine_displacement_cc': 1500, 'euro_standard': 4},
            'SUVs': {'class': 2, 'occupancy_min': 4.9, 'occupancy_max': 5.5, 'occupancy_avg': 5.2,
                     'fuel_consumption_l_per_km_free_flow': 0.12, 'fuel_consumption_l_per_km_congested': 0.18,
                     'fuel_type': 'petrol', 'weight_kg': 1800, 'engine_displacement_cc': 2000, 'euro_standard': 4},
            'Sedans': {'class': 2, 'occupancy_min': 4.9, 'occupancy_max': 5.5, 'occupancy_avg': 5.2,
                       'fuel_consumption_l_per_km_free_flow': 0.09, 'fuel_consumption_l_per_km_congested': 0.135,
                       'fuel_type': 'petrol', 'weight_kg': 1300, 'engine_displacement_cc': 1600, 'euro_standard': 4},
            'Wagons': {'class': 2, 'occupancy_min': 4.9, 'occupancy_max': 5.5, 'occupancy_avg': 5.2,
                       'fuel_consumption_l_per_km_free_flow': 0.11, 'fuel_consumption_l_per_km_congested': 0.165,
                       'fuel_type': 'petrol', 'weight_kg': 1400, 'engine_displacement_cc': 1700, 'euro_standard': 4},
            'Short Buses': {'class': 2, 'occupancy_min': 4.9, 'occupancy_max': 5.5, 'occupancy_avg': 5.2,
                            'fuel_consumption_l_per_km_free_flow': 0.15, 'fuel_consumption_l_per_km_congested': 0.225,
                            'fuel_type': 'petrol', 'weight_kg': 2500, 'engine_displacement_cc': 2500,
                            'euro_standard': 3},
            'Minibusses': {'class': 3, 'occupancy_min': 20.2, 'occupancy_max': 22.4, 'occupancy_avg': 21.3,
                           'fuel_consumption_l_per_km_free_flow': 0.2, 'fuel_consumption_l_per_km_congested': 0.3,
                           'fuel_type': 'petrol', 'weight_kg': 3500, 'engine_displacement_cc': 3000,
                           'euro_standard': 3},
            'Long Buses': {'class': 3, 'occupancy_min': 20.2, 'occupancy_max': 22.4, 'occupancy_avg': 21.3,
                           'fuel_consumption_l_per_km_free_flow': 0.25, 'fuel_consumption_l_per_km_congested': 0.375,
                           'fuel_type': 'petrol', 'weight_kg': 5000, 'engine_displacement_cc': 4000,
                           'euro_standard': 3},
            'Truck': {'class': 4, 'occupancy_min': 1.5, 'occupancy_max': 1.6, 'occupancy_avg': 1.55,
                      'fuel_consumption_l_per_km_free_flow': 0.3, 'fuel_consumption_l_per_km_congested': 0.45,
                      'fuel_type': 'diesel', 'weight_kg': 8000, 'engine_displacement_cc': 6000, 'euro_standard': 3},
            'Tanker and Trailer': {'class': 5, 'occupancy_min': 1.4, 'occupancy_max': 1.5, 'occupancy_avg': 1.45,
                                   'fuel_consumption_l_per_km_free_flow': 0.4,
                                   'fuel_consumption_l_per_km_congested': 0.6, 'fuel_type': 'diesel',
                                   'weight_kg': 15000, 'engine_displacement_cc': 8000, 'euro_standard': 2}
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
                'Nyanya Road': {
                    'Motorcycles': 100,
                    'Cars': 200,
                    'SUVs': 150,
                    'Sedans': 180,
                    'Wagons': 50,
                    'Short Buses': 30,
                    'Minibusses': 20,
                    'Long Buses': 10,
                    'Truck': 15,
                    'Tanker and Trailer': 5
                },
                'Lugbe Road': {
                    'Motorcycles': 80,
                    'Cars': 250,
                    'SUVs': 120,
                    'Sedans': 200,
                    'Wagons': 60,
                    'Short Buses': 25,
                    'Minibusses': 15,
                    'Long Buses': 8,
                    'Truck': 20,
                    'Tanker and Trailer': 10
                },
                'Kubwa Road': {
                    'Motorcycles': 120,
                    'Cars': 180,
                    'SUVs': 100,
                    'Sedans': 150,
                    'Wagons': 40,
                    'Short Buses': 20,
                    'Minibusses': 10,
                    'Long Buses': 5,
                    'Truck': 25,
                    'Tanker and Trailer': 8
                }
            }
            self.data = pd.DataFrame([
                {
                    'Date': self.analysis_date,
                    'Time': self.analysis_time,
                    'Road': road,
                    'Vehicle Type': vehicle,
                    'Real_Vehicle_Count': count,
                    'Real_VOR': self.vehicle_parameters[vehicle]['occupancy_avg'],
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
                    'Emission_Model': self.emission_model.value
                }
                for road in self.road_data
                for vehicle, count in self.road_data[road].items()
            ])
            logger.info("Loaded hardcoded data successfully")
        except Exception as e:
            logger.error(f"Error loading hardcoded data: {str(e)}")
            self.road_data = {}
            self.data = pd.DataFrame()

    def load_csv_data(self, csv_file_path: str):
        """Load traffic data from CSV file."""
        try:
            self.data = pd.read_csv(csv_file_path)
            self.road_data = {}
            for road in self.data['Road'].unique():
                road_df = self.data[self.data['Road'] == road]
                self.road_data[road] = {
                    vehicle: row['Real_Vehicle_Count']
                    for _, row in road_df.iterrows()
                    for vehicle in [row['Vehicle Type']]
                }
            logger.info(f"Loaded data from {csv_file_path}")
        except Exception as e:
            logger.error(f"Error loading CSV data from {csv_file_path}: {str(e)}")
            self.data = pd.DataFrame()
            self.road_data = {}

    def get_vehicle_distribution(self, road_name: str) -> List[Dict[str, Any]]:
        """Get vehicle distribution for a specific road."""
        try:
            if road_name not in self.road_data:
                return []
            return [
                {'vehicle_type': vehicle, 'count': count}
                for vehicle, count in self.road_data[road_name].items()
                if count > 0
            ]
        except Exception as e:
            logger.error(f"Error getting vehicle distribution for {road_name}: {str(e)}")
            return []

    def _calculate_barth_fuel_consumption(self, row):
        """Calculate fuel consumption using Barth's comprehensive model with configurable parameters"""
        vehicle_type = row['Vehicle Type']
        count = row['Real_Vehicle_Count']
        distance = row['Distance_KM']
        congested_time = row['Congested_Travel_Time_Minutes'] / 60  # Convert to hours

        # Get Barth parameters from row data or use defaults
        alpha = row.get('Barth_Alpha', self.BARTH_COEFFICIENTS['alpha'])
        beta = row.get('Barth_Beta', self.BARTH_COEFFICIENTS['beta'])
        gamma = row.get('Barth_Gamma', self.BARTH_COEFFICIENTS['gamma'])
        traffic_flow = row.get('Barth_Traffic_Flow', self.BARTH_COEFFICIENTS['traffic_flow'])
        road_gradient = row.get('Barth_Road_Gradient', self.BARTH_COEFFICIENTS['road_gradient'])
        acceleration_coef = row.get('Barth_Acceleration', self.BARTH_COEFFICIENTS['acceleration'])

        params = self.vehicle_parameters.get(vehicle_type, {})
        if not params:
            return 0, 0

        # Get vehicle parameters
        weight = params.get('weight_kg', 1000)
        engine_displacement = params.get('engine_displacement_cc', 1500)

        # Calculate average speed
        avg_speed = distance / congested_time if congested_time > 0 else 0

        # Barth's comprehensive formula
        # FC = (α * engine_displacement + β * weight + γ * gradient) * distance * traffic_flow
        #    + acceleration_coef * (number_of_stops * acceleration_events)

        # Estimate number of acceleration events based on congestion level
        congestion_factor = (45 - 4) / 45  # Based on delay time (simplified)
        acceleration_events = congestion_factor * 10  # Estimate 10 acceleration events per km in heavy congestion

        # Calculate fuel consumption
        base_consumption = (alpha * engine_displacement + beta * weight + gamma * road_gradient) * distance
        traffic_effect = base_consumption * traffic_flow
        acceleration_effect = acceleration_coef * acceleration_events * distance

        total_fuel_consumption = (base_consumption + traffic_effect + acceleration_effect) * count

        # For free flow (simplified)
        free_flow_consumption = (alpha * engine_displacement + beta * weight) * distance * 0.7 * count

        return free_flow_consumption, total_fuel_consumption

    def calculate_barth_emissions(self, vehicle_type: str, count: int, distance: float,
                                  speed: float, acceleration: float, idle_time: float) -> float:
        """
        Calculate emissions using Barth's model.

        Barth's model: Emissions = α * engine_displacement + β * weight + γ * gradient +
                              traffic_flow * speed + acceleration_factor * acceleration +
                              idle_factor * idle_time
        """
        try:
            vehicle_params = self.vehicle_parameters[vehicle_type]
            alpha = self.BARTH_COEFFICIENTS['alpha']
            beta = self.BARTH_COEFFICIENTS['beta']
            gamma = self.BARTH_COEFFICIENTS['gamma']
            traffic_flow = self.BARTH_COEFFICIENTS['traffic_flow']
            acceleration_factor = self.BARTH_COEFFICIENTS['acceleration']

            # Assume a default road gradient of 0% (flat road)
            road_gradient = 0.0

            # Calculate base emissions using Barth's formula
            base_emissions = (
                    alpha * vehicle_params['engine_displacement_cc'] +
                    beta * vehicle_params['weight_kg'] +
                    gamma * road_gradient +
                    traffic_flow * speed +
                    acceleration_factor * acceleration
            )

            # Add idle emissions
            idle_emissions = 0
            if idle_time > 0:
                if vehicle_params['fuel_type'] == 'petrol':
                    idle_emissions = self.BARTH_IDLE_FUEL_CONSUMPTION_PETROL * idle_time * 60
                else:
                    idle_emissions = self.BARTH_IDLE_FUEL_CONSUMPTION_DIESEL * idle_time * 60

            # Total emissions for this vehicle type
            total_emissions = (base_emissions * distance + idle_emissions) * count

            return total_emissions
        except Exception as e:
            logger.error(f"Error calculating Barth emissions for {vehicle_type}: {str(e)}")
            return 0.0

    def calculate_moves_emissions(self, vehicle_type: str, count: int, distance: float,
                                  speed: float, acceleration: float) -> float:
        """
        Calculate emissions using MOVES-like model.

        MOVES model: Emissions = base_rate * (1 + speed_correction * speed) *
                             (1 + accel_correction * acceleration) * distance
        """
        try:
            vehicle_params = self.vehicle_parameters[vehicle_type]

            if vehicle_params['fuel_type'] == 'petrol':
                base_rate = self.MOVES_BASE_EMISSION_RATE_PETROL
            else:
                base_rate = self.MOVES_BASE_EMISSION_RATE_DIESEL

            # Apply speed and acceleration corrections
            speed_correction = self.MOVES_SPEED_CORRECTION_FACTOR * speed
            accel_correction = self.MOVES_ACCEL_CORRECTION_FACTOR * acceleration

            # Calculate emissions
            emissions = base_rate * (1 + speed_correction) * (1 + accel_correction) * distance * count

            return emissions
        except Exception as e:
            logger.error(f"Error calculating MOVES emissions for {vehicle_type}: {str(e)}")
            return 0.0

    def calculate_basic_emissions(self, vehicle_type: str, count: int, distance: float) -> float:
        """
        Calculate emissions using basic model (fuel consumption * emission factor).
        """
        try:
            vehicle_params = self.vehicle_parameters[vehicle_type]

            # Get fuel consumption
            fuel_consumption = vehicle_params['fuel_consumption_l_per_km_congested'] * distance

            # Get appropriate emission factor
            if vehicle_params['fuel_type'] == 'petrol':
                emission_factor = self.EMISSION_FACTOR_PETROL
            else:
                emission_factor = self.EMISSION_FACTOR_DIESEL

            # Calculate emissions
            emissions = fuel_consumption * emission_factor * count

            return emissions
        except Exception as e:
            logger.error(f"Error calculating basic emissions for {vehicle_type}: {str(e)}")
            return 0.0

    def analyze_all_roads(self) -> Dict[str, Any]:
        """Analyze traffic data for all roads."""
        try:
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
                total_vehicles = road_data['Real_Vehicle_Count'].sum()
                total_people = sum(
                    row['Real_Vehicle_Count'] * self.vehicle_parameters[row['Vehicle Type']]['occupancy_avg']
                    for _, row in road_data.iterrows()
                )

                # Calculate fuel consumption using Barth model
                total_excess_fuel = 0
                total_fuel_cost = 0

                for _, row in road_data.iterrows():
                    # Get free flow and congested fuel consumption from Barth model
                    free_flow_fuel, congested_fuel = self._calculate_barth_fuel_consumption(row)

                    # Calculate excess fuel (difference between congested and free flow)
                    excess_fuel = congested_fuel - free_flow_fuel
                    total_excess_fuel += excess_fuel

                    # Calculate fuel cost
                    vehicle_params = self.vehicle_parameters[row['Vehicle Type']]
                    if vehicle_params['fuel_type'] == 'petrol':
                        fuel_cost = excess_fuel * self.FUEL_PRICE_PETROL
                    else:
                        fuel_cost = excess_fuel * self.FUEL_PRICE_DIESEL

                    total_fuel_cost += fuel_cost

                # Calculate CO2 emissions based on selected model
                total_co2 = 0.0
                for _, row in road_data.iterrows():
                    vehicle_type = row['Vehicle Type']
                    count = row['Real_Vehicle_Count']

                    if self.emission_model == EmissionModelType.BARTH:
                        # Use Barth model with additional parameters
                        speed = row.get('Congested_Speed_KHP', 8.0)
                        acceleration = row.get('Avg_Acceleration', 0.5)
                        idle_time = row.get('Idle_Time_Percentage', 0.3) * row.get('Congested_Travel_Time_Minutes',
                                                                                   45.0) / 60.0
                        total_co2 += self.calculate_barth_emissions(vehicle_type, count, self.CORRIDOR_LENGTH,
                                                                    speed, acceleration, idle_time)
                    elif self.emission_model == EmissionModelType.MOVES:
                        # Use MOVES model
                        speed = row.get('Congested_Speed_KPH', 8.0)
                        acceleration = row.get('Avg_Acceleration', 0.5)
                        total_co2 += self.calculate_moves_emissions(vehicle_type, count, self.CORRIDOR_LENGTH,
                                                                    speed, acceleration)
                    else:
                        # Use basic model
                        total_co2 += self.calculate_basic_emissions(vehicle_type, count, self.CORRIDOR_LENGTH)

                # Calculate productivity loss
                total_productivity_loss = total_people * (road_data['Congested_Travel_Time_Minutes'].iloc[0] -
                                                          road_data['Free_Flow_Time_Minutes'].iloc[
                                                              0]) * self.VALUE_PER_MINUTE

                road_results[road] = {
                    'total_vehicles': total_vehicles,
                    'total_people': total_people,
                    'total_excess_fuel_l': total_excess_fuel,
                    'total_co2_kg': total_co2,
                    'total_fuel_cost_naira': total_fuel_cost,
                    'total_productivity_loss_naira': total_productivity_loss
                }

                total_summary['total_vehicles_all_roads'] += total_vehicles
                total_summary['total_people_all_roads'] += total_people
                total_summary['total_excess_fuel_all_roads'] += total_excess_fuel
                total_summary['total_co2_all_roads'] += total_co2
                total_summary['total_fuel_cost_all_roads'] += total_fuel_cost
                total_summary['total_productivity_loss_all_roads'] += total_productivity_loss
                total_summary['total_delay_hours_all_roads'] += (road_data['Congested_Travel_Time_Minutes'].iloc[0] -
                                                                 road_data['Free_Flow_Time_Minutes'].iloc[
                                                                     0]) / 60.0 * total_people

            return {
                'road_results': road_results,
                'total_summary': total_summary
            }
        except Exception as e:
            logger.error(f"Error analyzing roads: {str(e)}")
            return {'road_results': {}, 'total_summary': {}}

    def generate_report(self) -> pd.DataFrame:
        """Generate a detailed report DataFrame."""
        try:
            results = self.analyze_all_roads()
            data = []
            for road, road_data in results['road_results'].items():
                data.append({
                    'Road': road,
                    'Vehicle Count': road_data['total_vehicles'],
                    'People Affected': road_data['total_people'],
                    'Excess Fuel (L)': road_data['total_excess_fuel_l'],
                    'CO2 Emissions (kg)': road_data['total_co2_kg'],
                    'Fuel Cost (Naira)': road_data['total_fuel_cost_naira'],
                    'Productivity Loss (Naira)': road_data['total_productivity_loss_naira'],
                    'Total Economic Impact (Naira)': road_data['total_fuel_cost_naira'] + road_data[
                        'total_productivity_loss_naira'],
                    'Emission Model': self.emission_model.value
                })
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return pd.DataFrame()

    def get_available_models(self) -> List[str]:
        """Return list of available emission models."""
        return [model.value for model in EmissionModelType]

    def generate_charts(self, output_dir: str = "static/charts") -> Dict[str, str]:
        """
        Generate and save pie and bar charts for the report.
        Returns a dictionary of chart file paths.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            chart_files = {}

            # Generate pie charts for vehicle distribution per road
            for road_name in self.road_data.keys():
                vehicle_dist = self.get_vehicle_distribution(road_name)
                if not vehicle_dist:
                    logger.warning(f"No vehicle distribution data for {road_name}")
                    continue

                vehicle_types = [item["vehicle_type"] for item in vehicle_dist]
                counts = [item["count"] for item in vehicle_dist]

                plt.figure(figsize=(6, 4))
                plt.pie(counts, labels=vehicle_types, autopct='%1.1f%%')
                plt.title(f"{road_name} Vehicle Distribution")
                pie_chart_path = os.path.join(output_dir, f"pie_{road_name.replace(' ', '_')}.png")
                plt.savefig(pie_chart_path, format="png", bbox_inches="tight")
                plt.close()
                chart_files[f"pie_{road_name.replace(' ', '_')}"] = pie_chart_path
                logger.info(f"Saved pie chart for {road_name} at {pie_chart_path}")

            # Generate bar charts for summary metrics
            summary = self.analyze_all_roads()['total_summary']
            roads = list(self.road_data.keys())

            # Bar chart: People Affected
            people_data = [self.analyze_all_roads()['road_results'][road]['total_people'] for road in roads]
            plt.figure(figsize=(6, 4))
            plt.bar(roads, people_data)
            plt.title("People Affected by Congestion")
            plt.xlabel("Road")
            plt.ylabel("Number of People")
            plt.xticks(rotation=45)
            people_chart_path = os.path.join(output_dir, "people_chart.png")
            plt.savefig(people_chart_path, format="png", bbox_inches="tight")
            plt.close()
            chart_files["people_chart"] = people_chart_path
            logger.info(f"Saved people chart at {people_chart_path}")

            # Bar chart: Excess Fuel Consumption
            fuel_data = [self.analyze_all_roads()['road_results'][road]['total_excess_fuel_l'] for road in roads]
            plt.figure(figsize=(6, 4))
            plt.bar(roads, fuel_data)
            plt.title("Excess Fuel Consumption")
            plt.xlabel("Road")
            plt.ylabel("Liters")
            plt.xticks(rotation=45)
            fuel_chart_path = os.path.join(output_dir, "fuel_chart.png")
            plt.savefig(fuel_chart_path, format="png", bbox_inches="tight")
            plt.close()
            chart_files["fuel_chart"] = fuel_chart_path
            logger.info(f"Saved fuel chart at {fuel_chart_path}")

            # Bar chart: CO2 Emissions
            co2_data = [self.analyze_all_roads()['road_results'][road]['total_co2_kg'] for road in roads]
            plt.figure(figsize=(6, 4))
            plt.bar(roads, co2_data)
            plt.title("Excess CO₂ Emissions")
            plt.xlabel("Road")
            plt.ylabel("kg")
            plt.xticks(rotation=45)
            co2_chart_path = os.path.join(output_dir, "co2_chart.png")
            plt.savefig(co2_chart_path, format="png", bbox_inches="tight")
            plt.close()
            chart_files["co2_chart"] = co2_chart_path
            logger.info(f"Saved CO2 chart at {co2_chart_path}")

            # Bar chart: Total Cost
            cost_data = [
                self.analyze_all_roads()['road_results'][road]['total_fuel_cost_naira'] +
                self.analyze_all_roads()['road_results'][road]['total_productivity_loss_naira']
                for road in roads
            ]
            plt.figure(figsize=(6, 4))
            plt.bar(roads, cost_data)
            plt.title("Total Cost (Fuel + Productivity Loss)")
            plt.xlabel("Road")
            plt.ylabel("Naira")
            plt.xticks(rotation=45)
            cost_chart_path = os.path.join(output_dir, "cost_chart.png")
            plt.savefig(cost_chart_path, format="png", bbox_inches="tight")
            plt.close()
            chart_files["cost_chart"] = cost_chart_path
            logger.info(f"Saved cost chart at {cost_chart_path}")

            logger.info(f"Generated charts and saved to {output_dir}")
            return chart_files
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            return {}