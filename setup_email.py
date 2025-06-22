#!/usr/bin/env python3
"""
Email Configuration Setup and Test Script

This script helps users set up and test email functionality for the Meeting Summarizer.
It will:
1. Guide users through setting up their email credentials
2. Save the credentials to a .env file
3. Optionally test the email configuration

Usage:
    python setup_email.py
"""

import os
import sys
import getpass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def test_smtp_connection(server, port, username, password, from_email):
    """Test SMTP connection."""
    try:
        with smtplib.SMTP(server, port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(username, password)
            logger.info("SMTP connection successful!")
            return True
    except Exception as e:
        logger.error(f"SMTP connection failed: {e}")
        return False

def send_test_email(server, port, username, password, from_email, to_email):
    """Send a test email."""
    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = "Meeting Analyzer - Email Test"
        message["From"] = from_email
        message["To"] = to_email
        
        text = """
        This is a test email from the Meeting Analyzer.
        
        If you're receiving this email, your email configuration is working correctly!
        
        You can now use the email functionality in the Meeting Analyzer.
        """
        
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                h1 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }
                .success { color: #27ae60; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Meeting Analyzer - Email Test</h1>
                <p>This is a test email from the Meeting Analyzer.</p>
                <p class="success">If you're receiving this email, your email configuration is working correctly!</p>
                <p>You can now use the email functionality in the Meeting Analyzer.</p>
            </div>
        </body>
        </html>
        """
        
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        
        message.attach(part1)
        message.attach(part2)
        
        with smtplib.SMTP(server, port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(username, password)
            server.send_message(message)
            logger.info(f"Test email sent to {to_email}")
            return True
    except Exception as e:
        logger.error(f"Failed to send test email: {e}")
        return False

def save_env_file(config):
    """Save configuration to .env file."""
    env_path = Path('.env')
    
    # Check if file exists and if we should overwrite
    if env_path.exists():
        overwrite = input("A .env file already exists. Overwrite? (y/n): ").lower().strip() == 'y'
        if not overwrite:
            logger.info("Aborted. Existing .env file not modified.")
            return False
    
    # Write the configuration
    with open(env_path, 'w') as f:
        f.write("# Email configuration for Meeting Analyzer\n")
        for key, value in config.items():
            f.write(f"{key}={value}\n")
    
    logger.info(f"Configuration saved to {env_path.absolute()}")
    return True

def main():
    """Main function."""
    print("\n" + "="*60)
    print("Meeting Analyzer - Email Configuration Setup")
    print("="*60 + "\n")
    
    print("This script will help you set up email functionality for the Meeting Analyzer.")
    print("You'll need to provide your SMTP server details and credentials.\n")
    
    # Get email configuration
    config = {}
    
    # SMTP Server
    config["SMTP_SERVER"] = input("SMTP Server (e.g., smtp.gmail.com): ").strip()
    if not config["SMTP_SERVER"]:
        logger.error("SMTP Server cannot be empty.")
        return
    
    # SMTP Port
    port = input("SMTP Port (default: 587): ").strip()
    config["SMTP_PORT"] = port if port else "587"
    try:
        int(config["SMTP_PORT"])
    except ValueError:
        logger.error("Port must be a number.")
        return
    
    # SMTP Username
    config["SMTP_USERNAME"] = input("SMTP Username (your email): ").strip()
    if not validate_email(config["SMTP_USERNAME"]):
        logger.error("Invalid email format for SMTP Username.")
        return
    
    # SMTP Password
    print("\nFor Gmail, you'll need to use an App Password instead of your regular password.")
    print("You can generate one at https://myaccount.google.com/apppasswords\n")
    config["SMTP_PASSWORD"] = getpass.getpass("SMTP Password: ")
    if not config["SMTP_PASSWORD"]:
        logger.error("Password cannot be empty.")
        return
    
    # From Email
    default_from = config["SMTP_USERNAME"]
    from_email = input(f"From Email (default: {default_from}): ").strip()
    config["EMAIL_FROM"] = from_email if from_email else default_from
    
    # Test connection
    print("\nTesting SMTP connection...")
    if not test_smtp_connection(
        config["SMTP_SERVER"], 
        int(config["SMTP_PORT"]), 
        config["SMTP_USERNAME"], 
        config["SMTP_PASSWORD"],
        config["EMAIL_FROM"]
    ):
        print("\nConnection test failed. Please check your settings and try again.")
        retry = input("Do you want to save these settings anyway? (y/n): ").lower().strip() == 'y'
        if not retry:
            return
    
    # Save configuration
    if save_env_file(config):
        print("\nEmail configuration saved successfully!")
        
        # Ask to send test email
        send_test = input("\nDo you want to send a test email? (y/n): ").lower().strip() == 'y'
        if send_test:
            to_email = input("Enter recipient email address: ").strip()
            if validate_email(to_email):
                print(f"Sending test email to {to_email}...")
                send_test_email(
                    config["SMTP_SERVER"], 
                    int(config["SMTP_PORT"]), 
                    config["SMTP_USERNAME"], 
                    config["SMTP_PASSWORD"],
                    config["EMAIL_FROM"],
                    to_email
                )
            else:
                logger.error("Invalid recipient email address.")
    
    print("\nSetup complete!")
    print("\nTo use these settings in your application, you can:")
    print("1. Load the .env file using a library like python-dotenv")
    print("2. Set the environment variables manually before running the application")
    print("\nExample with python-dotenv:")
    print("```python")
    print("from dotenv import load_dotenv")
    print("load_dotenv()  # This will load the variables from .env")
    print("```")

if __name__ == "__main__":
    main() 