import sys
import os
import requests
import numpy as np
import matplotlib.pyplot as plt


# 텔레그램 알림
def telegram_noti(text):
    bot_id = 'bot324447573:AAHwi1JMc_opfGMtgKzLyQt9O_eINy6vlco'
    my_id = '338332660'
    telegram_payload = {
        'chat_id': my_id,
        'text': text
    }
    requests.get('https://api.telegram.org/' + bot_id + '/sendMessage', params=telegram_payload)


def db2normal(a):
    return 10 ** (a / 10)


def normal2db(a):
    return 10 * np.log10(a)


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


# 폴더 생성 함수
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# plot
def plotCurrentResult(env):
    plt.figure(0)
    plt.subplot2grid((10, 9), (0, 0), colspan=3, rowspan=4)
    plt.plot(env.system.avg_pwr)
    plt.subplot2grid((10, 9), (4, 0), colspan=3, rowspan=6)
    # plt.matshow(env.state)

    for i in range(env.user_no):
        plt.subplot2grid((10, 9), ((i // 6)*2, 3 + i % 6))
        plt.plot(env.system.avg_rate[i])
        plt.subplot2grid((10, 9), ((i // 6)*2+1, 3 + i % 6))
        plt.plot(env.system.mu[i])

    plt.show(block=False)
