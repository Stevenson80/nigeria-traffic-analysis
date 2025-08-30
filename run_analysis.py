#!/usr/bin/env python3
"""
Standalone script to run Abuja traffic analysis without the web interface
Generates comprehensive reports suitable for PDF output
"""

import pandas as pd
import logging
import os
import sys
from datetime import datetime

# Add the parent directory to the path to import traffic_analysis_models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_analysis_models import TrafficAnalysisModel

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
    """Format numbers with commas"""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return "0"


def format_currency(value):
    """Format currency values"""
    try:
        return f"₦{float(value):,.2f}"
    except (ValueError, TypeError):
        return "₦0.00"


def run_analysis(csv_file="traffic_data.csv"):
    """Run traffic analysis and generate reports

    Args:
        csv_file (str): Path to the CSV file with traffic data

    Returns:
        dict: Analysis results including summary and detailed data
    """
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            logger.warning(f"CSV file {csv_file} not found. Using hardcoded data.")

        # Load data from CSV or use hardcoded data
        model = TrafficAnalysisModel(csv_file_path=csv_file)

        # Run analysis
        logger.info("Running traffic analysis...")
        results = model.analyze_all_roads()

        # Generate detailed report
        report_df = model.generate_report()

        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'abuja_traffic_analysis_report_{timestamp}.csv'
        report_df.to_csv(report_filename, index=False)
        logger.info(f"Report saved to {report_filename} with {len(report_df)} rows")

        # Print summary
        summary = results['total_summary']
        print("\n" + "=" * 60)
        print("           ABUJA TRAFFIC ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Source: {csv_file if os.path.exists(csv_file) else 'Hardcoded Data'}")
        print(f"Total Vehicles: {format_number(summary.get('total_vehicles_all_roads', 0))}")
        print(f"Total People Affected: {format_number(summary.get('total_people_all_roads', 0))}")
        print(f"Total Delay: {summary.get('total_delay_hours_all_roads', 0.0):,.1f} hours")
        print(f"Total Excess Fuel: {summary.get('total_excess_fuel_all_roads', 0.0):,.1f} liters")
        print(f"Total Fuel Cost: {format_currency(summary.get('total_fuel_cost_all_roads', 0.0))}")
        print(f"Total CO2 Emissions: {summary.get('total_co2_all_roads', 0.0):,.1f} kg")
        print(f"Total Productivity Loss: {format_currency(summary.get('total_productivity_loss_all_roads', 0.0))}")
        print(
            f"Total Economic Impact: {format_currency(summary.get('total_fuel_cost_all_roads', 0.0) + summary.get('total_productivity_loss_all_roads', 0.0))}")

        # Print per-road results
        print("\n" + "=" * 60)
        print("               PER-ROAD RESULTS")
        print("=" * 60)
        for road_name, road_data in results['road_results'].items():
            print(f"\n{road_name.upper()}:")
            print(f"  Vehicles: {format_number(road_data.get('total_vehicles', 0))}")
            print(f"  People: {format_number(road_data.get('total_people', 0))}")
            print(f"  Excess Fuel: {road_data.get('total_excess_fuel_l', 0.0):,.1f} L")
            print(f"  Fuel Cost: {format_currency(road_data.get('total_fuel_cost_naira', 0.0))}")
            print(f"  CO2 Emissions: {road_data.get('total_co2_kg', 0.0):,.1f} kg")
            print(f"  Productivity Loss: {format_currency(road_data.get('total_productivity_loss_naira', 0.0))}")
            economic_impact = road_data.get('total_fuel_cost_naira', 0.0) + road_data.get(
                'total_productivity_loss_naira', 0.0)
            print(f"  Total Economic Impact: {format_currency(economic_impact)}")

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Detailed report: {report_filename}")
        print("=" * 60)

        return {
            'summary': summary,
            'road_results': results['road_results'],
            'report_df': report_df,
            'report_filename': report_filename
        }

    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}")
        print(f"ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    # Allow specifying CSV file as command line argument
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "traffic_data.csv"

    print("Abuja Traffic Analysis - Standalone Mode")
    print("=" * 40)

    try:
        results = run_analysis(csv_file)
        # You could add PDF generation here using the returned results
        # generate_pdf_report(results)

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        sys.exit(1)