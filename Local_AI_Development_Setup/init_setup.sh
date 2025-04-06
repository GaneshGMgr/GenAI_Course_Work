# Script for Virtual Environm
echo "[$(date)]: START"
echo "[$(date)]: Creating venv with Python"
python -m venv venv
echo "[$(date)]: Activating venv"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "[$(date)]: Upgrading pip"
python -m pip install --upgrade pip
echo "[$(date)]: Installing requirements"
pip install -r requirements.txt
echo "[$(date)]: END"

# # # Script for Conda Environment Creation
# echo "[$(date)]: START"
# echo "[$(date)]: Creating Conda environment"
# conda create --name venv -y
# echo "[$(date)]: Activating Conda environment"
# if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
#     conda activate venv
# else
#     source activate venv
# fi
# echo "[$(date)]: Upgrading pip"
# python -m pip install --upgrade pip
# echo "[$(date)]: Installing requirements"
# pip install -r requirements.txt
# echo "[$(date)]: END"


## bash init_setup.sh # to run this .sh file in terminal Using Git Bash
