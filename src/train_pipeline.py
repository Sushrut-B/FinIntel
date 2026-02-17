import subprocess
import logging
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training_pipeline.log"),
        logging.StreamHandler()
    ]
)

# Gmail SMTP configuration - temporarily commented out due to auth issues
# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587
# SMTP_USERNAME = "1dt23ai058@dsatm.edu.in"
# SMTP_PASSWORD = "xyz@0058"  # Replace with your Gmail App Password
# EMAIL_FROM = SMTP_USERNAME
# EMAIL_TO = SMTP_USERNAME

# def send_email(subject, body):
#     msg = MIMEMultipart()
#     msg['From'] = EMAIL_FROM
#     msg['To'] = EMAIL_TO
#     msg['Subject'] = subject
#     msg.attach(MIMEText(body, 'plain'))
#     try:
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls()
#             server.login(SMTP_USERNAME, SMTP_PASSWORD)
#             server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
#         logging.info("Notification email sent successfully.")
#     except Exception as e:
#         logging.error(f"Failed to send notification email: {e}")

def run_command(cmd):
    logging.info(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        error_msg = f"Command failed with exit code {result.returncode}: {cmd}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    logging.info(f"Command completed successfully: {cmd}")

def main():
    try:
        run_command("python src/models/train_baseline.py")
        run_command("python src/models/train_tft.py")
        run_command("python src/models/run_anomaly_detection.py")

        success_msg = "Training pipeline completed successfully."
        logging.info(success_msg)
        # Temporarily skip email notification due to SMTP errors
        # send_email("Training Pipeline Success", success_msg)

    except Exception as e:
        error_msg = f"Training pipeline failed: {e}"
        logging.error(error_msg)
        # send_email("Training Pipeline Failure", error_msg)
        raise

if __name__ == "__main__":
    main()
