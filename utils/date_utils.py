from datetime import datetime


def get_time_str():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S").replace(r'/', '_').replace(r' ', '_').replace(r':', '_')
