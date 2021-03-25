#nohup $1 10 7 1 0 > srv1.txt 2>&1 &
#nohup $1 10 8 1 1 > srv2.txt 2>&1 &
#nohup $1 10 7 1 2 > srv3.txt 2>&1 &
#nohup $1 10 8 1 3 > srv4.txt 2>&1 &

nohup $1 100000 500000 1 0 > srv1.txt 2>&1 &
nohup $1 100000 500000 1 1 > srv2.txt 2>&1 &
nohup $1 100000 500000 1 2 > srv3.txt 2>&1 &
nohup $1 100000 500000 1 3 > srv4.txt 2>&1 &
