#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#DOCUMENTATION: https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html

from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()
@sched.scheduled_job(trigger='cron', day_of_week='sun', start_date='2024-03-31',end_date='2025-03-31',
                     hour=16, minute=54, second=35) #Cron-based scheduling

#the job that you would like to schedule must be defined in a function.
def scheduled_job(): 
    print("Scheduling job is started!!!")

sched.start()

