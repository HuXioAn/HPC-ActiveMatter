import matplotlib.pyplot as plt
import numpy as np
import re


def main():


    with open('output.plot', 'r') as file:
        content = file.read()

    # 提取基本参数
    parameters = re.search(r'generalParameter\{fieldLength=(\d+\.\d+),totalStep=(\d+),birdNum=(\d+)\}', content)
    field_length = float(parameters.group(1))
    total_step = int(parameters.group(2))
    bird_num = int(parameters.group(3))


    posX = np.zeros(bird_num)
    posY = np.zeros(bird_num)
    theta = np.zeros(bird_num)


    fig = plt.figure(figsize=(4,4), dpi=80)
    ax = plt.gca()

    data_lines = content.split('\n')[1:]  # 跳过第一行
    for i, line in enumerate(data_lines):
        line = line.strip('{};')
        data_points = line.split(';')
        for j, point in enumerate(data_points):
            (posX[j], posY[j], theta[j]) = point.split(',')

        plt.cla()
        plt.quiver(posX, posY, np.cos(theta), np.sin(theta))
        ax.set(xlim=(0, field_length), ylim=(0, field_length))
        ax.set_aspect('equal')	
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.pause(0.001)

    plt.show()


    return 0


if __name__== "__main__":
    main()