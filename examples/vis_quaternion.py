from boardAttitude import utils
if __name__ == "__main__":
#    fname = './results/result_q_2.txt'
    fname = './results/result_q.txt'
#    fname = './results/result_angle.txt'
#    fname = './results/record_angle.txt'
    box_roll = utils.BoxRoll(fname,useQuat=True)
    box_roll.show()
