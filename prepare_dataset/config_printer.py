"""
Configuration printing utilities for dataset preparation.
"""

from tabulate import tabulate


class DatasetPreparationConfigPrinter:
    """Pretty printer for dataset preparation configuration."""
    
    def __init__(self, config_dict):
        self.config_dict = config_dict
    
    def print_table(self):
        """Print configuration as a formatted table."""
        table_data = [[key, value] for key, value in self.config_dict.items()]
        print("\n" + "="*60)
        print("DATASET PREPARATION CONFIGURATION")
        print("="*60)
        print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))
        print("="*60 + "\n")
