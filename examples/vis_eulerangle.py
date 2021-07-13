from boardAttitude import utils
import numpy as np
if __name__ == "__main__":
#    fname = './results/result_angle_q.txt'
    fname = './results/result_angle.txt'
#    fname = './results/record_angle.txt'
    rot_order = 'XYZ'
    box_roll = utils.BoxRoll(fname,useQuat=False,rot_order=rot_order)
    box_roll.show(fps=50)
