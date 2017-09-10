# deprecated, use standard logging instead

from argute.util import UtilManager
import datetime

f = open(UtilManager.log_file, 'a')
f.writelines("========== start logging ========== " )

def log(stmt):
    f.writelines(stmt)
