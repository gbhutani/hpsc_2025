# Fix for SMTP Authentication Error with Gmail

## Issue
When attempting to send emails via Gmail's SMTP server using Python's `smtplib`, authentication fails with error messages such as:
```
smtplib.SMTPAuthenticationError: (535, b'5.7.8 Username and Password not accepted.')
smtplib.SMTPAuthenticationError: (534, b'5.7.9 Application-specific password required.')
```

## Solution
### Step 1: Enable 2-Step Verification
1. Go to [Google Account Security](https://myaccount.google.com/security).
2. Under **"Signing in to Google"**, enable **2-Step Verification**.
3. Follow the on-screen steps to complete the setup.

### Step 2: Generate an App-Specific Password
1. Go to [Google App Passwords](https://myaccount.google.com/apppasswords).
2. Select **Mail** as the app and **Other (Custom Name)** for the device.
3. Generate the password and copy it. This will be used instead of your normal password.

### Step 3: Update Your Python Script
Modify your Python script to use the new **App Password** and enable `starttls`.

```python
import smtplib

email = "your-email@gmail.com"
password = "your-app-password"  # Use the generated App Password

try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(email, password)
    print("Login successful!")
    server.quit()
except smtplib.SMTPAuthenticationError as e:
    print("Authentication failed:", e)
except Exception as e:
    print("Other error:", e)
```

### Step 4: Test the Script
Run the script to verify the authentication works:
```
$ python3 test_login.py
Login successful!
```

Now, your script should send emails without authentication errors. 🚀