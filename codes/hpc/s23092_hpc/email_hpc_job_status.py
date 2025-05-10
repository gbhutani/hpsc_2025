import os, subprocess
import time
import sys

import smtplib
from email.mime.text import MIMEText

started_flag = False
completed_flag = False
send_started_email = False
hpc_path='/home/s23092/HPSC_ME522'
# hpc_path = sys.argv[2]
jobid='567986'
# jobid = sys.argv[1]
hpc_userid = 's23092'  # insert your hpc user id here

started_file_name = jobid + '.n199.cluster.iitmandi.ac.in.started'
completed_file_name = jobid + '.n199.cluster.iitmandi.ac.in.completed'
cmd_copy_started_file = f'scp {hpc_userid}@10.8.1.19:{hpc_path}/{started_file_name} "D:/I-PhD 2022/HPC/hpsc_2025/codes/hpc/s23092_hpc/"'
cmd_copy_completed_file = f'scp {hpc_userid}@10.8.1.19:{hpc_path}/{completed_file_name} "D:/I-PhD 2022/HPC/hpsc_2025/codes/hpc/s23092_hpc/"'


def send_email(type):
    types_allowed = ['started', 'completed']
    assert type in types_allowed, "type not allowed in send_email()"

    myEmail = 'siddharthamithiya.acad2001@gmail.com'
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()

    # password.txt contains the email password or app password if Google 2-step verification is on
    with open(r'D:\I-PhD 2022\HPC\Notes\random_letters.txt', 'r') as file:
        password = file.read()
        password = password.strip()
    server.login(myEmail, password)
    print('logged in')

    if type == 'started':
        print(f"{jobid} started on HPC")
        with open(started_file_name, 'r') as file:
            contents = file.read()
        msg = MIMEText(contents)
        msg['From'] = myEmail
        msg['To'] = myEmail
        msg['Subject'] = jobid + ' started'
        server.send_message(msg)
    if type == 'completed':
        print(f"{jobid} completed on HPC")
        with open(completed_file_name, 'r') as file:
            contents = file.read()
        msg = MIMEText(contents)
        msg['From'] = myEmail
        msg['To'] = myEmail
        msg['Subject'] = jobid + ' completed'
        server.send_message(msg)
    # send email
    server.quit()


while True:

    if not started_flag:
        subprocess.call(cmd_copy_started_file, shell=True, stderr=subprocess.DEVNULL)
        print('cmd_copy_started_file')
        if os.path.exists(started_file_name):
            print('trying to send email')
            send_email('started')
            started_flag = True
    elif not completed_flag:
        subprocess.call(cmd_copy_completed_file, shell=True)
        if os.path.exists(completed_file_name):
            send_email('completed')
            completed_flag = True
    else:
        break
    time.sleep(5)
