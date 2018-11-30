import threading, time
import random

def start_process_user(user_id, cp_time):
    t = threading.Thread(target=process_user, args=(user_id, cp_time))
    t.start()
    # t.join()
    print('next')

def process_user(user_id, cp_time):
    # print(cp_time)
    print('start process user %d' % (user_id))
    time.sleep(cp_time)
    print('end of user %d' % user_id)

for i in range(10):
    start_process_user(i, random.randrange(1, 10))
