from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time():
    korea_timezone = ZoneInfo('Asia/Seoul')
    now_kst = datetime.now(korea_timezone)
    formatted_time = now_kst.strftime("%m%d%H%M")
    return formatted_time
