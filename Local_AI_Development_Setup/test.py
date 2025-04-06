# single_file_test_example.py

import pytest

# Function to check if a number is even
def is_even(number: int) -> bool:
    """Check if a number is even."""
    if number % 2 == 0:
        return True
    return False

# Function to add two numbers
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Test for the 'is_even' function
def test_is_even():
    assert is_even(2) == True  # Even number
    assert is_even(3) == False  # Odd number
    assert is_even(0) == True  # Zero is considered even


# Run the tests when this script is executed
if __name__ == "__main__":
    pytest.main()
