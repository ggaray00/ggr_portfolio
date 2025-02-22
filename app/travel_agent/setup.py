# Environment and library setup

import os
import getpass

def setup_env(var):
    if not os.getenv(var):
        os.environ[var] = getpass.getpass(f'Enter value for {var}: ')
