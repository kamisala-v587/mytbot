# 容器什么时候启动的
ps -p 1 -o lstart,cmd

Fri Jul  3 06:19:06 2026 /package/admin/s6/command/s6-svscan -d4 -- /run/service

# 训练什么时候开始写的 log
grep "Start offline" outputs/TBot_SA1/pretrain_v1/2026-07-03/01-46-07_\ /train.log | head -1


```
root@vla-reborn-0:~/vla/workspace/mytbot# RUN="/home/jovyan/vla/workspace/mytbot/outputs/TBot_SA1/pretrain_v1/2026-07-03/01-46-07_ "

echo "开训:"
grep "Start offline" "${RUN}/train.log" | head -1

echo "最后step:"
grep "lerobot_train.py:step" "${RUN}/train.log" | grep "06:17" | tail -1 | cut -c1-60

echo "容器启动:"
ps -p 1 -o lstart=

tail -1 "${RUN}/loss.log"
开训:
INFO 2026-07-03 01:58:02 ot_train.py:998 Start offline training on a fixed dataset
最后step:
INFO 2026-07-03 06:17:50 lerobot_train.py:step 04:1
容器启动:
Fri Jul  3 06:19:06 2026
9850,2026-07-03 06:17:50,0.001565,0.005553,25.070160,121.069898,128.880859,131.292969,100.000000,0.660030,0.080767,0.335878,0.064146,1.326272,0.000050,0.205618,1.550233
root@vla-reborn-0:~/vla/workspace/mytbot# 


```

### 本次指纹记录

root@vla-reborn-0:~# date | tee ~/container_fingerprint.txt
Fri Jul  3 06:56:44 AM UTC 2026
root@vla-reborn-0:~# hostname | tee -a ~/container_fingerprint.txt
vla-reborn-0
root@vla-reborn-0:~# ps -p 1 -o lstart= | tee -a ~/container_fingerprint.txt
Fri Jul  3 06:19:06 2026
root@vla-reborn-0:~# 


后续检测：
root@vla-reborn-0:~# ps -p 1 -o lstart=
Fri Jul  3 06:19:06 2026

执行 ps -p 1 -o lstart= 检测容器开始时间，如果和上次时间一致，说明容器没有被重启过