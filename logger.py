import logging
import os
from pathlib import Path

# Define the path to the log file
log_dir = Path("logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "pipeline.log")

# Configure logging settings
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a logger instance
logger = logging.getLogger(__name__)

# Optionally, set up console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
