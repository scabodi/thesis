#!C:\Users\sara\PycharmProjects\thesis\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'easydev==0.9.38','console_scripts','browse'
__requires__ = 'easydev==0.9.38'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('easydev==0.9.38', 'console_scripts', 'browse')()
    )
