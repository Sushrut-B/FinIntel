import subprocess
import logging

logging.basicConfig(
    filename="scheduled_training.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def main():
    try:
        logging.info("Starting training pipeline...")
        result = subprocess.run(
            "python src/train_pipeline.py",
            shell=True,
            capture_output=True,
            text=True
        )
        logging.info(result.stdout)
        if result.returncode != 0:
            logging.error(f"Training pipeline failed with error: {result.stderr}")
        else:
            logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Exception during training pipeline execution: {e}")

if __name__ == "__main__":
    main()
