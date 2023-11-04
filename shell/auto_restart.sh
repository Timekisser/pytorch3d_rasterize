#!/bin/sh
 
#添加本地执行路径
export LD_LIBRARY_PATH=./
 
while true; do
        #启动一个循环，定时检查进程是否存在
        server=`ps aux | grep gen_objaverse_pointcloud.sh | grep -v grep`
        if [ ! "$server" ]; then
            #如果不存在就重新启动
            ./shell/gen_objaverse_pointcloud.sh
            #启动后沉睡10s
            sleep 1
            echo "restart"
        fi
        #每次循环沉睡10s
        sleep 10
        echo "sleep 10"
done
