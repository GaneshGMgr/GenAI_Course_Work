from setuptools import find_packages, setup

setup(
    name="Local_AI_Development_Setup",
    version="0.0.1",
    author="QASystem",
    author_email="qasystem@gmail.com",
    packages=find_packages(include=["QASystem", "QASystem.*"]),  # Include all subpackages of QASystem
    install_requires=[], 
    include_package_data=True,  # Ensures non-Python files like templates are included
    package_data={  # Includes templates or other static files
        "QASystem": ["templates/*.html"],
    },
    entry_points={  # Command-line tool setup
        'console_scripts': [
            'qasystem=app:main',  # Ensure app.py has a main() function
        ],
    },
)
