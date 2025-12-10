"""
Runner script for the Mutual Funds ML Project (Risk Prediction Model)

Usage:
    python runner.py train     -> Train BEST RISK MODEL (train_risk_model.py)
    python runner.py serve     -> Start FastAPI server (serve.py)
    python runner.py all       -> Train model + start server
"""

import sys
import subprocess
import os

# Ensure working directory is the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_train():
    """Run the risk model training script."""
    print("\n🔄 Training BEST Risk Prediction Model...\n")
    try:
        subprocess.run([sys.executable, "train_risk_model.py"], check=True)
    except subprocess.CalledProcessError as e:
        print("\n❌ Training failed!")
        print("Details:", e)
        sys.exit(1)

    print("\n✅ Risk model training completed. Saved: models/risk_model.joblib\n")


def run_serve():
    """Start FastAPI API server."""
    print("\n🚀 Starting FastAPI Risk Prediction Server...\n")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m", "uvicorn",
                "serve:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000",
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("\n❌ ERROR starting FastAPI server!")
        print("Details:", e)
        sys.exit(1)


def run_all():
    """Train the risk model first, then launch the API."""
    run_train()
    run_serve()


def print_help():
    print("""
📘 Mutual Funds ML Project — Runner Script

Usage:
    python runner.py train     -> Train ONLY the Risk Prediction Model
    python runner.py serve     -> Run ONLY FastAPI model server
    python runner.py all       -> Train + Start FastAPI server

Examples:
    python runner.py train
    python runner.py serve
    python runner.py all
""")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_help()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        run_train()
    elif command == "serve":
        run_serve()
    elif command == "all":
        run_all()
    else:
        print_help()
