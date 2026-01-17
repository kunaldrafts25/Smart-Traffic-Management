
# Lazy imports to avoid dependency issues at module load time
__all__ = [
    'ModernVehicleDetector',
]

def __getattr__(name):
    if name == 'ModernVehicleDetector':
        from .core.modern_vehicle_detector import ModernVehicleDetector
        return ModernVehicleDetector
    raise AttributeError(f"module 'src' has no attribute '{name}'")
