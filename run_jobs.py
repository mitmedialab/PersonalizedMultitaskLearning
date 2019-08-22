""" This file allows multiple jobs to be run on a server. After each job, an 
	email is sent to notify desired people of its completion.

	Must specify a text job file that contains the names and commands for each 
	job. Each job has 4 lines, containing: 
		1) the name, 
		2) the command, 
		3) the location of a file where the job output should be saved, 
		4) a blank line.

	An example job file format is as follows:

		Job1
		python job.py path1 arg1
		path/output1.txt

		Job2
		python job.py path2 arg2
		path/output2.txt

	Usage: python run_jobs.py jobs.txt
"""

import os
import sys
import smtplib
import string
from time import time
import helperFuncs as helper

DEFAULT_EMAIL_LIST = ['myemail@gmail.com', 'youremail@gmail.com']
SENDING_ADDRESS = 'myemail@gmail.com'
MINIMUM_JOB_SECONDS = 600 # 10 minutes
PRINT_LAST_X_LINES = 300
ERROR = 1
SUCCESS = 0
WARNING = 2


def reload_files():
	reload(helper)

class Job:
	def __init__(self, name, command, output_file):
		self.name = name
		self.command = command
		self.output_file = output_file.rstrip('\n')


def send_email(subject, text, to_addr_list=DEFAULT_EMAIL_LIST):
	body = string.join(('From: %s' % SENDING_ADDRESS,
						'To: %s' % to_addr_list,
						'Subject: %s' % subject,
						'',
						text), '\r\n')

	try:
		server = smtplib.SMTP('smtp.gmail.com:587')  # NOTE:  This is the GMAIL SSL port.
		server.ehlo() # this line was not required in a previous working version
		server.starttls()
		server.login(SENDING_ADDRESS, 'gmail_password')
		server.sendmail(SENDING_ADDRESS, to_addr_list, body)
		server.quit()
		print "Email sent successfully!"
	except:
		return "Email failed to send!"

def load_job_file(filename):
	f = open(filename, 'r')
	lines = f.readlines()

	jobs = []

	i = 0
	while i < len(lines):
		jobname = lines[i]
		command = lines[i+1]
		output_file = lines[i+2]
		job = Job(jobname, command, output_file)
		jobs.append(job)
		i = i+4

	return jobs

def run_job(job_obj):
	""" Runs a system command for a job, returns whether it 
		succeeded and output text to be emailed.

		Inputs:
			job_obj: an instance of the Job class

		Returns
			A code indicating whether the job was successful, and
			a string containing text about the job and job output to 
			be mailed to the user
	"""

	print "\nRunning job", job_obj.name	
	
	if os.path.exists(job_obj.output_file):
		message = "The desired output file " + job_obj.output_file + " already exists."
		print "Error!", message
		return ERROR, message

	t0 = time()

	# execute the command
	stream = os.popen(job_obj.command)
	output = stream.read()

	# save output to desired file
	of = open(job_obj.output_file, 'w')
	of.write(output)
	of.close()

	t1 = time()
	total_secs = t1 - t0

	hours, mins, secs = helper.get_secs_mins_hours_from_secs(total_secs)
	time_str = "Job ended. Total time taken: " + str(int(hours)) + "h " + str(int(mins)) + "m " + str(int(secs)) + "s"
	print time_str

	if not os.path.exists(job_obj.output_file):
		message = "Job failed to create the desired output file."
		print "Error!", message
		code = ERROR
	elif total_secs < MINIMUM_JOB_SECONDS:
		message = "The total time taken for the job was suspiciously short."
		print "Warning!", message
		code = WARNING
	else:
		message = ""
		print "Job finished successfully!"
		code = SUCCESS

	lines = output.split('\n')
	tail = "\n".join(lines[-PRINT_LAST_X_LINES:])

	message += "\n\n" + time_str + "\n\n"
	message += "The last " + str(PRINT_LAST_X_LINES) + " lines of job output were:\n\n"
	message += tail

	return code, message

def email_about_job(job_obj, status, output):
	if status == ERROR:
		title = "Error! Problem with job " + job_obj.name
	elif status == SUCCESS:
		title = "Success! Job " + job_obj.name + " is finished"
	else:
		title = "Warning! Job " + job_obj.name + " finished too quickly"

	send_email(title, output)

def run_jobs(jobfile):
	jobs = load_job_file(filename)

	for job in jobs:
		status, output = run_job(job)
		email_about_job(job, status, output)

	send_email("ALL JOBS FINISHED!!", "Congratulations, all of the jobs in the file " + jobfile + " have finished running.")

if __name__ == "__main__":
	if len(sys.argv) < 1:
		print "Error! Usage is python run_jobs.py jobs.txt"
		print "See this file's documentation for required format for jobs.txt"

	filename= sys.argv[1]
	jobfile=sys.argv[1]
	print "Running all jobs in file", jobfile, ". . ."

	run_jobs(jobfile)
