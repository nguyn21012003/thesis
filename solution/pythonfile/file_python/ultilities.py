from datetime import datetime

import os


def saveFunction(currentProgram, motherFolder):

    date = datetime.now()
    print(date.strftime("%a-%m-%Y"))

    directory = f"/motherFolder/date/currentProgram/"

    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory
