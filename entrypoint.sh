#!/bin/bash

echo "Welcome! Please choose an option:"
echo "1. Run the training script for the first task (Xgboost with Optuna)"
echo "2. Start the Jupyter Notebook for evaluation of the 1st task (if browser is not detected, please use the last link to open the notebook)"
echo "3. Run the training script for the second task (multimodal training)"
echo "4. Start the Jupyter Notebook for evaluation of the 2nd task"
echo "5. Start the Jupyter Notebook for evaluation"
echo "6. Inspect the code (open a shell in the code directory)"
read -p "Enter your choice (1, 2, 3, 4, 5, or 6): " choice

if [ "$choice" -eq 1 ]; then
    echo "Running training script..."
    python3 src/1st_task/training.py

elif [ "$choice" -eq 2 ]; then 
    echo "Starting Jupyter Notebook..."
    
    # Capture full output and redirect both stdout and stderr 
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root > /tmp/jupyter_notebook_log.txt 2>&1 &
    
    # Get the PID of the Jupyter Notebook process
    JUPYTER_PID=$!
    
    # Wait a bit to ensure the process starts
    sleep 5
    
    # Check if the process is running
    if kill -0 $JUPYTER_PID 2>/dev/null; then
        echo "Jupyter Notebook process started with PID: $JUPYTER_PID"
        
        # Try to extract token from log file
        TOKEN=$(grep -oP 'token=\K[a-zA-Z0-9]+' /tmp/jupyter_notebook_log.txt | head -n 1)
        
        if [ -n "$TOKEN" ]; then
            NOTEBOOK_PATH="src/1st_task/notebooks/tabular_evaluation.ipynb"
            NOTEBOOK_URL="http://localhost:8888/notebooks/${NOTEBOOK_PATH}?token=${TOKEN}"
            
            echo "========================================"
            echo "Notebook URL: $NOTEBOOK_URL"
            echo "Please copy and paste this URL into your web browser"
            echo "========================================"
        else
            echo "Could not extract token from Jupyter Notebook log"
        fi
    else
        echo "Failed to start Jupyter Notebook process"
    fi
    
    wait

elif [ "$choice" -eq 3 ]; then
    echo "Running training script..."
    python src/2nd_task/main.py

elif [ "$choice" -eq 4 ]; then
    echo "Starting Jupyter Notebook..."
    
    # Capture full output and redirect both stdout and stderr 
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root > /tmp/jupyter_notebook_log.txt 2>&1 &
    
    # Get the PID of the Jupyter Notebook process
    JUPYTER_PID=$!
    
    # Wait a bit to ensure the process starts
    sleep 5
    
    # Check if the process is running
    if kill -0 $JUPYTER_PID 2>/dev/null; then
        echo "Jupyter Notebook process started with PID: $JUPYTER_PID"
        
        # Try to extract token from log file
        TOKEN=$(grep -oP 'token=\K[a-zA-Z0-9]+' /tmp/jupyter_notebook_log.txt | head -n 1)
        
        if [ -n "$TOKEN" ]; then
            NOTEBOOK_PATH="src/2nd_task/notebooks/multimodal_eval.ipynb"
            NOTEBOOK_URL="http://localhost:8888/notebooks/${NOTEBOOK_PATH}?token=${TOKEN}"
            
            echo "========================================"
            echo "Notebook URL: $NOTEBOOK_URL"
            echo "Please copy and paste this URL into your web browser"
            echo "========================================"
        else
            echo "Could not extract token from Jupyter Notebook log"
        fi
    else
        echo "Failed to start Jupyter Notebook process"
    fi
    
    wait

elif [ "$choice" -eq 5 ]; then 
    echo "Starting Jupyter Notebook..."
    
    # Capture full output and redirect both stdout and stderr 
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root > /tmp/jupyter_notebook_log.txt 2>&1 &
    
    # Get the PID of the Jupyter Notebook process
    JUPYTER_PID=$!
    
    # Wait a bit to ensure the process starts
    sleep 5
    
    # Check if the process is running
    if kill -0 $JUPYTER_PID 2>/dev/null; then
        echo "Jupyter Notebook process started with PID: $JUPYTER_PID"
        
        # Try to extract token from log file
        TOKEN=$(grep -oP 'token=\K[a-zA-Z0-9]+' /tmp/jupyter_notebook_log.txt | head -n 1)
        
        if [ -n "$TOKEN" ]; then
            NOTEBOOK_URL="http://localhost:8888/?token=${TOKEN}"
            
            echo "========================================"
            echo "Jupyter Notebook URL: $NOTEBOOK_URL"
            echo "Please copy and paste this URL into your web browser"
            echo "========================================"
        else
            echo "Could not extract token from Jupyter Notebook log"
        fi
    else
        echo "Failed to start Jupyter Notebook process"
    fi

    wait

elif [ "$choice" -eq 6 ]; then
    echo "Opening a shell in the code directory. You can inspect the code here."
    cd /candidate_challenge  # Navigate to the folder where everything is saved
    exec /bin/bash  # Start a shell in this directory

else
    echo "Invalid choice. Exiting."
    exit 1
fi
