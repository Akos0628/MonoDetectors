import numpy as np

class detectionInfo(object):
    def __init__(self, line):
        #print('line: name=%s truncated=%f occluded=%f alpha=%f\n bbox=[%f %f %f %f]\n dimensions=[%f %f %f]\n location=[%f %f %f]\n rotation_y=%f score=%f' % (line[0], npline[0], npline[1], npline[2], npline[3], npline[4], npline[5], npline[6], npline[7], npline[8], npline[9], npline[10], npline[11], npline[12], npline[13], npline[14]))

        self.name = line[0]

        self.truncation = np.float64(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = np.float64(line[3])

        # in pixel coordinate
        self.xmin = int(line[4])
        self.ymin = int(line[5])
        self.xmax = int(line[6])
        self.ymax = int(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = np.float64(line[8])
        self.w = np.float64(line[9])
        self.l = np.float64(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = np.float64(line[11])
        self.ty = np.float64(line[12])
        self.tz = np.float64(line[13])

        # global orientation [-pi, pi]
        self.rot_global = np.float64(line[14])

        # score
        self.score = np.float64(line[15])


def toArrayFromString(line):
    return line.strip().split(' ')

def predArrayToResult(preds):
    result = []
    for line in preds:
        result.append(toArrayFromString(line))

    return result

def stringFromLine(line):
    return f'{line[0]} {line[1]} {line[2]} {line[3]} {int(line[4])} {int(line[5])} {int(line[6])} {int(line[7])} {line[8]} {line[9]} {line[10]} {line[11]} {line[12]} {line[13]} {line[14]} {line[15]}'