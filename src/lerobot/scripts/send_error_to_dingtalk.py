import traceback
import urllib.request

RELAY_IP = "10.6.2.41" # 4090ti

def send_error_to_dingtalk(error_info):
    url = f"http://{RELAY_IP}:9999"
    data = error_info.encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "text/plain"})
    try:
        urllib.request.urlopen(req, timeout=10)
        print("[通知] 报错已发送到钉钉")
    except Exception as e:
        print(f"[通知失败] {e}")