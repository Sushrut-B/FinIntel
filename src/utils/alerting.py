import smtplib
from email.mime.text import MIMEText
import os

def send_anomaly_alert(anomalies_df, to_email):
    if anomalies_df.empty:
        return  # No anomalies to alert

    body = "UPI Macro Intelligence Alert:\n\nThe following anomalies were detected:\n\n"
    for _, row in anomalies_df.iterrows():
        body += f"Date: {row['ds'].strftime('%Y-%m-%d')}, Value: {row['y']}, Residual: {row['residual']:.2f}\n"
    
    msg = MIMEText(body)
    msg['Subject'] = 'UPI Macro Intelligence Anomaly Alert'
    msg['From'] = os.getenv("ALERT_EMAIL", "bankalgisushrut@gmail.com")  # Use environment variable if set
    msg['To'] = to_email

    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    ALERT_EMAIL = os.getenv("ALERT_EMAIL", "bankalgisushrut@gmail.com")
    ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD")  # Must be set in environment

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(ALERT_EMAIL, ALERT_EMAIL_PASSWORD)
            server.send_message(msg)
        print("Anomaly alert email sent.")
    except Exception as e:
        print(f"Failed to send anomaly alert email: {e}")
