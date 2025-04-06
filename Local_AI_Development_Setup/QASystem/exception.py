import sys
import logging

class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        # Capture error message and traceback details
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()  # Extract traceback details
        self.lineno = exc_tb.tb_lineno  # Line number where the exception occurred
        self.file_name = exc_tb.tb_frame.f_code.co_filename  # File where exception occurred

    def __str__(self):
        # String representation of the exception to include file name, line number, and error message
        return f"Error occurred in Python script [{self.file_name}] line number [{self.lineno}] error message [{self.error_message}]"

if __name__ == "__main__":
    try:
        a = 1 / 0  # This will cause a ZeroDivisionError
    except Exception as e:
        # Log the error before raising the custom exception
        logging.error(f"Exception occurred: {str(e)}")
        raise CustomException(str(e), sys)
