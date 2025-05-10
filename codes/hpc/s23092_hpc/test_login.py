import smtplib

email = "siddharthamithiya.acad2001@gmail.com"
password = "password"  # Use the 16-character App Password

try:
    server = smtplib.SMTP("smtp.gmail.com", 587)  # Connect to Gmail SMTP
    server.ehlo()  # Identify with the server
    server.starttls()  # Upgrade connection to TLS
    server.ehlo()
    server.login(email, password)  # Login with App Password
    print("Login successful!")
    server.quit()
except smtplib.SMTPAuthenticationError as e:
    print("Authentication failed:", e)
except Exception as e:
    print("Other error:", e)
