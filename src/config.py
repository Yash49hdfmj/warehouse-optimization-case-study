from dataclasses import dataclass
from typing import Dict


@dataclass
class OptimizerConfig:
    """Configuration for the warehouse optimizer"""
    target_days: int = 7
    service_level: float = 0.95
    cost_per_km: float = 15.0
    co2_per_ton_km: float = 62.0
    optimize_weight: str = "Balanced"  # Options: "Cost", "CO2", "Balanced"
    train_test_split: float = 0.8
    
    def validate(self):
        """Validate configuration parameters"""
        assert 0 < self.service_level <= 1.0, "Service level must be between 0 and 1"
        assert self.target_days > 0, "Target days must be positive"
        assert self.optimize_weight in ["Cost", "CO2", "Balanced"], "Invalid optimize_weight"


@dataclass
class CityLocation:
    """Geographic coordinates for cities"""
    name: str
    lat: float
    lon: float


# City coordinates for distance calculation
CITY_LOCATIONS = {
    "Mumbai": CityLocation("Mumbai", 19.0760, 72.8777),
    "Delhi": CityLocation("Delhi", 28.6139, 77.2090),
    "Bangalore": CityLocation("Bangalore", 12.9716, 77.5946),
    "Chennai": CityLocation("Chennai", 13.0827, 80.2707),
    "Kolkata": CityLocation("Kolkata", 22.5726, 88.3639),
    "Hyderabad": CityLocation("Hyderabad", 17.3850, 78.4867),
    "Pune": CityLocation("Pune", 18.5204, 73.8567),
    "Ahmedabad": CityLocation("Ahmedabad", 23.0225, 72.5714)
}