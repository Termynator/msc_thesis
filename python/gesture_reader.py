import numpy as np
import matplotlib.pyplot as plt
import slayerSNN as snn
from dv import LegacyAedatFile

path = 'path_to_aedat_files'

actionName = [
    'hand_clapping',
    'right_hand_wave',
    'left_hand_wave',
    'right_arm_clockwise',
    'right_arm_counter_clockwise',
    'left_arm_clockwise', 
    'left_arm_counter_clockwise',
    'arm_roll',
    'air_drums',
    'air_guitar',
    'other_gestures',
]

def readAedatEvent(filename):
    xEvent = []
    yEvent = []
    pEvent = []
    tEvent = []
    with LegacyAedatFile(filename) as f:
        for event in f:
            xEvent.append(event.x)
            yEvent.append(event.y)
            pEvent.append(event.polarity)
            tEvent.append(event.timestamp/1000)

    return xEvent, yEvent, pEvent, tEvent

def splitData(filename, path):
    x, y, p, t = scipy.io.loadmat(path + filename + '.aedat')
    labels = np.loadtxt(path + filename + '_labels.csv', delimiter=',', skiprows=1)
    labels[:,0]  -= 1
    labels[:,1:]

    if not os.path.isdir('data/' + filename):
        os.mkdir('data/' + filename)

    lastAction = 100
    for action, tst, ten in labels:
        if action == lastAction:    continue # This is to ignore second arm_roll samples
        print(actionName[int(action)])
        ind = (t >= tst) & (t < ten)
        TD = snn.io.event(x[ind], y[ind], p[ind], (t[ind] - tst)/1000)
        # snn.io.showTD(TD)
        lastAction = action

        snn.io.encodeNpSpikes('data/'+ filename + '/{:g}.npy'.format(action), TD)

if __name__ == '__main__':
    user = np.arange(29) + 1
    lighting = [
        'fluorescent',
        'fluorescent_led',
        'lab',
        'led',
        'natural',
    ]

    count = 0
    for id in user:
        for light in lighting:
            filename = 'user{:02d}_{}'.format(id, light)

            if os.path.isfile(path + filename + '.aedat'):
                print(count, filename)
                splitData(filename, path)
                count += 1
