import numpy
import matplotlib.pyplot as plt


def main():
    x_train = numpy.loadtxt('X_train.txt')

    y_train = numpy.loadtxt('y_train.txt')

    all_data = numpy.column_stack((x_train[:, 0], y_train))
    male_heigth = all_data[all_data[:, 1] == 0][:, 0]
    female_height = all_data[all_data[:, 1] == 1][:, 0]

    plt.hist(male_heigth, alpha=0.6, label='Male', color='blue')
    plt.hist(female_height, alpha=0.6, label='Female', color='pink')

    plt.xlabel('Height (cm)')
    plt.ylabel('Frequency')
    plt.title('Height Histogram by Gender')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
