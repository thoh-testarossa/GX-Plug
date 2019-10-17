cd ../build/bin
nohup ./srv_UtilServerTest_JumpIteration 400000 2000001 4 0 > srv1.txt 2>&1 &
nohup ./srv_UtilServerTest_JumpIteration 400000 2000000 4 1 > srv2.txt 2>&1 &
nohup ./srv_UtilServerTest_JumpIteration 400000 1999999 4 2 > srv3.txt 2>&1 &
nohup ./srv_UtilServerTest_JumpIteration 400000 2000000 4 3 > srv4.txt 2>&1 &
