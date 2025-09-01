#!/usr/bin/env python3
"""
Standalone script to run Abuja traffic analysis without the web interface.
Generates comprehensive reports (CSV and PDF) using the updated TrafficAnalysisModel,
supporting five vehicle classes: Motorcycles, Cars, Buses, Light Trucks, Heavy Trucks.
"""

import pandas as pd
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for importing traffic_analysis_models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_analysis_models import TrafficAnalysisModel, EmissionModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def format_number(value):
    """Format numbers with commas for readability."""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for number formatting: {value}")
        return "0"

def format_currency(value):
    """Format currency values with Naira symbol."""
    try:
        return f"₦{float(value):,.0f}"
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for currency formatting: {value}")
        return "₦0"

def format_float(value, decimals=1):
    """Format float values with specified decimal places."""
    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for float formatting: {value}")
        return f"0.{'0' * decimals}"

def generate_pdf_report(model, output_path=None):
    """
    Generate a PDF report using the TrafficAnalysisModel.
    The PDF includes a homepage and detailed analysis report.

    Args:
        model: TrafficAnalysisModel instance with analysis results.
        output_path (str, optional): Path for the PDF output file. If None, uses a timestamped filename.

    Returns:
        str: Path to the generated PDF file, or None if generation fails.
    """
    try:
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"abuja_traffic_analysis_report_{timestamp}.pdf"

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) or '.'
        os.makedirs(output_dir, exist_ok=True)

        # Generate PDF using the model's method
        pdf_path = model.generate_pdf_report(output_path)
        if pdf_path and os.path.exists(pdf_path):
            logger.info(f"PDF report generated successfully: {pdf_path}")
            return pdf_path
        else:
            logger.error(f"Failed to generate PDF report at {output_path}")
            return None
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
        return None

def validate_csv_file(csv_file):
    """
    Validate the CSV file and provide helpful error messages.

    Args:
        csv_file (str): Path to the CSV file

    Returns:
        bool: True if file is valid, False otherwise
    """
    if not csv_file:
        logger.info("No CSV file provided, expecting command-line arguments or hardcoded data")
        return False

    csv_path = Path(csv_file)
    if not csv_path.exists():
        logger.warning(f"CSV file {csv_file} not found. Expecting command-line arguments or hardcoded data.")
        return False

    if csv_path.stat().st_size == 0:
        logger.warning(f"CSV file {csv_file} is empty. Expecting command-line arguments or hardcoded data.")
        return False

    try:
        # Try to read the CSV to check if it's valid
        test_df = pd.read_csv(csv_file)
        if test_df.empty:
            logger.warning(f"CSV file {csv_file} contains no data. Expecting command-line arguments or hardcoded data.")
            return False

        required_columns = ['Road', 'Motorcycle_Count', 'Car_Count', 'Bus_Count', 'Light_Truck_Count', 'Heavy_Truck_Count']
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            logger.warning(f"CSV file missing required columns: {missing_columns}. Expecting command-line arguments or hardcoded data.")
            return False

        return True
    except Exception as e:
        logger.warning(f"Error reading CSV file {csv_file}: {str(e)}. Expecting command-line arguments or hardcoded data.")
        return False

def run_analysis(csv_file=None, generate_pdf=True, emission_model="basic",
                motorcycles=0, cars=0, buses=0, light_trucks=0, heavy_trucks=0):
    """
    Run traffic analysis and generate CSV and PDF reports.

    Args:
        csv_file (str): Path to the CSV file with traffic data.
        generate_pdf (bool): Whether to generate a PDF report.
        emission_model (str): Emission model to use ('basic', 'barth', or 'moves').
        motorcycles (int): Number of motorcycles (used if no CSV).
        cars (int): Number of cars (used if no CSV).
        buses (int): Number of buses (used if no CSV).
        light_trucks (int): Number of light trucks (used if no CSV).
        heavy_trucks (int): Number of heavy trucks (used if no CSV).

    Returns:
        dict: Analysis results including summary, road results, report DataFrame, and file paths.
    """
    try:
        # Validate CSV file
        use_csv = validate_csv_file(csv_file)

        # Validate command-line vehicle volumes if no CSV is used
        vehicle_volumes = {
            'Motorcycles': motorcycles,
            'Cars': cars,
            'Buses': buses,
            'Light Trucks': light_trucks,
            'Heavy Trucks': heavy_trucks
        }
        if not use_csv:
            for vehicle_type, volume in vehicle_volumes.items():
                try:
                    volume = int(volume)
                    if volume < 0:
                        logger.error(f"Invalid {vehicle_type} volume: {volume}. Must be non-negative.")
                        raise ValueError(f"Invalid {vehicle_type} volume: {volume}. Must be non-negative.")
                except (ValueError, TypeError):
                    logger.error(f"Invalid {vehicle_type} volume: {volume}. Must be a non-negative integer.")
                    raise ValueError(f"Invalid {vehicle_type} volume: {volume}. Must be a non-negative integer.")

        # Map emission model string to EmissionModelType
        emission_model_map = {
            'basic': EmissionModelType.BASIC,
            'barth': EmissionModelType.BARTH,
            'moves': EmissionModelType.MOVES
        }
        if emission_model not in emission_model_map:
            logger.error(f"Invalid emission model: {emission_model}. Defaulting to 'basic'.")
            emission_model = 'basic'

        # Initialize model
        logger.info(f"Initializing TrafficAnalysisModel with emission model: {emission_model}")
        if use_csv:
            model = TrafficAnalysisModel(
                csv_file_path=csv_file,
                emission_model=emission_model_map[emission_model]
            )
        else:
            # Pass vehicle volumes as a dictionary to the model
            model = TrafficAnalysisModel(
                csv_file_path=None,
                emission_model=emission_model_map[emission_model],
                vehicle_volumes=vehicle_volumes
            )

        # Check if we have data
        if model.data.empty:
            logger.error("No data available for analysis")
            raise ValueError("No traffic data available for analysis")

        # Run analysis
        logger.info("Starting traffic analysis...")  # Fixed: Changed _logger to logger
        results = model.analyze_all_roads()
        if not results or not results['road_results']:
            logger.error("Analysis returned no results.")
            raise ValueError("Traffic analysis failed to produce results.")

        # Generate detailed report
        report_df = model.generate_report()
        if report_df.empty:
            logger.error("Generated report is empty.")
            raise ValueError("Report generation produced no data.")

        # Save CSV report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'abuja_traffic_analysis_report_{timestamp}.csv'
        report_df.to_csv(report_filename, index=False)
        logger.info(f"CSV report saved to {report_filename} with {len(report_df)} rows")

        # Generate PDF report if requested
        pdf_filename = None
        if generate_pdf:
            pdf_filename = generate_pdf_report(model, f"abuja_traffic_analysis_report_{timestamp}.pdf")
            if not pdf_filename:
                logger.warning("PDF generation failed, but analysis completed.")

        # Extract summary for console output
        summary = results.get('total_summary', {})
        if not summary:
            logger.error("No summary data found in analysis results.")
            raise ValueError("Summary data is missing.")

        # Print summary to console
        print("\n" + "=" * 60)
        print("           ABUJA TRAFFIC ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Source: {csv_file if use_csv else 'Command-line or Hardcoded Data'}")
        print(f"Emission Model: {model.emission_model.value.upper()}")
        print(f"Total Vehicles: {format_number(summary.get('total_vehicles_all_roads', 0))}")
        print(f"  Motorcycles: {format_number(summary.get('total_motorcycles', 0))}")
        print(f"  Cars: {format_number(summary.get('total_cars', 0))}")
        print(f"  Buses: {format_number(summary.get('total_buses', 0))}")
        print(f"  Light Trucks: {format_number(summary.get('total_light_trucks', 0))}")
        print(f"  Heavy Trucks: {format_number(summary.get('total_heavy_trucks', 0))}")
        print(f"Total People Affected: {format_number(summary.get('total_people_all_roads', 0))}")
        print(f"Total Delay: {format_float(summary.get('total_delay_hours_all_roads', 0.0), 1)} hours")
        print(f"Total Excess Fuel: {format_float(summary.get('total_excess_fuel_all_roads', 0.0), 1)} liters")
        print(f"Total Fuel Cost: {format_currency(summary.get('total_fuel_cost_all_roads', 0.0))}")
        print(f"Total CO2 Emissions: {format_float(summary.get('total_co2_all_roads', 0.0), 1)} kg")
        print(f"Total Productivity Loss: {format_currency(summary.get('total_productivity_loss_all_roads', 0.0))}")
        total_economic_impact = summary.get('total_fuel_cost_all_roads', 0.0) + \
                                summary.get('total_productivity_loss_all_roads', 0.0)
        print(f"Total Economic Impact: {format_currency(total_economic_impact)}")

        # Print per-road results
        road_results = results.get('road_results', {})
        if road_results:
            print("\n" + "=" * 60)
            print("               PER-ROAD RESULTS")
            print("=" * 60)
            for road_name, road_data in road_results.items():
                print(f"\n{road_name.upper()}:")
                print(f"  Vehicles: {format_number(road_data.get('total_vehicles', 0))}")
                print(f"    Motorcycles: {format_number(road_data.get('total_motorcycles', 0))}")
                print(f"    Cars: {format_number(road_data.get('total_cars', 0))}")
                print(f"    Buses: {format_number(road_data.get('total_buses', 0))}")
                print(f"    Light Trucks: {format_number(road_data.get('total_light_trucks', 0))}")
                print(f"    Heavy Trucks: {format_number(road_data.get('total_heavy_trucks', 0))}")
                print(f"  People: {format_number(road_data.get('total_people', 0))}")
                print(f"  Excess Fuel: {format_float(road_data.get('total_excess_fuel_l', 0.0), 1)} L")
                print(f"  Fuel Cost: {format_currency(road_data.get('total_fuel_cost_naira', 0.0))}")
                print(f"  CO2 Emissions: {format_float(road_data.get('total_co2_kg', 0.0), 1)} kg")
                print(f"  Productivity Loss: {format_currency(road_data.get('total_productivity_loss_naira', 0.0))}")
                economic_impact = road_data.get('total_fuel_cost_naira', 0.0) + \
                                  road_data.get('total_productivity_loss_naira', 0.0)
                print(f"  Total Economic Impact: {format_currency(economic_impact)}")

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Detailed CSV report: {report_filename}")
        if pdf_filename:
            print(f"Comprehensive PDF report: {pdf_filename}")
        print("=" * 60)

        return {
            'summary': summary,
            'road_results': road_results,
            'report_df': report_df,
            'report_filename': report_filename,
            'pdf_filename': pdf_filename,
            'model': model
        }

    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}", exc_info=True)
        print(f"ERROR: {str(e)}")
        raise

def main():
    """Main function to handle command line execution."""
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run Abuja Traffic Analysis with updated vehicle classes')
    parser.add_argument('--csv-file', default=None,
                        help='Path to the CSV file with traffic data (Road, Motorcycle_Count, Car_Count, Bus_Count, Light_Truck_Count, Heavy_Truck_Count)')
    parser.add_argument('--no-pdf', action='store_true',
                        help='Skip PDF generation (only generate CSV report)')
    parser.add_argument('--emission-model', choices=['basic', 'barth', 'moves'], default='basic',
                        help='Emission model to use for calculations')
    parser.add_argument('--list-models', action='store_true',
                        help='List available emission models and exit')
    parser.add_argument('--motorcycles', type=int, default=0,
                        help='Number of motorcycles (used if no CSV file)')
    parser.add_argument('--cars', type=int, default=0,
                        help='Number of cars (used if no CSV file)')
    parser.add_argument('--buses', type=int, default=0,
                        help='Number of buses (used if no CSV file)')
    parser.add_argument('--light-trucks', type=int, default=0,
                        help='Number of light trucks (used if no CSV file)')
    parser.add_argument('--heavy-trucks', type=int, default=0,
                        help='Number of heavy trucks (used if no CSV file)')

    args = parser.parse_args()

    if args.list_models:
        print("Available emission models:")
        print("  - basic: Simple fuel consumption-based emission calculations")
        print("  - barth: Barth's comprehensive fuel consumption model")
        print("  - moves: EPA MOVES-like emission model")
        return

    print("Abuja Traffic Analysis - Standalone Mode")
    print("=" * 40)

    try:
        # Run analysis with specified parameters
        results = run_analysis(
            csv_file=args.csv_file,
            generate_pdf=not args.no_pdf,
            emission_model=args.emission_model,
            motorcycles=args.motorcycles,
            cars=args.cars,
            buses=args.buses,
            light_trucks=args.light_trucks,
            heavy_trucks=args.heavy_trucks
        )

        # Return success exit code
        return 0

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())