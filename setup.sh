#!/bin/bash

# Function to display an error message and exit
function error_exit {
    echo "$1" 1>&2
    exit 1
}

# Create a virtual environment
python3 -m venv env || error_exit "Failed to create virtual environment"

# Activate the virtual environment
source env/bin/activate || error_exit "Failed to activate virtual environment"

# Clone the repository
git clone https://github.com/holoviz-topics/panel-chat-examples || error_exit "Failed to clone a dependency repository"

# Navigate into the directory
cd panel-chat-examples || error_exit "Failed to navigate into the repository directory"

# Install the package with all optional dependencies
pip install -e ".[all]" || error_exit "Failed to install the package with optional dependencies"

# Navigate back to the root directory
cd .. || error_exit "Failed to navigate back to the root directory"

# Install other dependencies from requirements.txt if it exists
if [ -f requirements.txt ]; then
    pip install -r requirements.txt || error_exit "Failed to install dependencies from requirements.txt"
fi

echo "Setup complete. Don't forget to set your OPENAI_API_KEY environment variable and export, it is needed. Also activate your env"
