import logging
import os
from datetime import datetime

# Create log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"  # Use dashes and full year

# Set log directory path
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

# Combine log directory path with log file name
LOGFILEPATH = os.path.join(log_path, LOG_FILE)

# Configure the logger
logging.basicConfig(
    level=logging.INFO, 
    filename=LOGFILEPATH, 
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Added space after timestamp
    datefmt="%Y-%m-%d %H:%M:%S"  # Date format for timestamp
)

# Log an example message
# logging.info("This is a test log message")
