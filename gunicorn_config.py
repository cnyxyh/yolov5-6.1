#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 监听本机的端口
bind = "0.0.0.0:7000"
# 未决连接的最大数量，即等待服务的客户的数量
backlog = 2048
# 进程数
workers = 21
# 线程数
threads = 2
# 工作模式为gevent
# worker_class = 'gevent'
worker_class = 'gevent'
# 最大客户端并发数量，默认情况下这个值为1000。
worker_connections = 1000
# 超时 默认30秒
timeout = 120
# 连接上等待请求的秒数，默认情况下值为2
keepalive = 50
# 根目录,server.py所在目录
chdir = '/'
# 记录PID
# pidfile = 'gunicorn.pid'


#gunicorn -k uvicorn.workers.UvicornWorker --bind "0.0.0.0:8080" --log-level debug main:app
# gunicorn -c gunicorn_config.py main:app

# gunicorn -k uvicorn.workers.UvicornWorker -c gunicorn_config.py main:app