import numpy
import matplotlib.pyplot as plt

def main():

    x_data = numpy.loadtxt('male_female_X_train.txt')
    y_data = numpy.loadtxt('male_female_y_train.txt')
    all_data = numpy.column_stack((x_data[:, 0], y_data))
    male_heigth = all_data[all_data[:, 1] == 0][:, 0]
    female_heigth = all_data[all_data[:, 1] == 1][:, 0]

    plt.hist(male_heigth, alpha=0.6, label='Male', color='blue')
    plt.hist(female_heigth, alpha=0.6, label='Female', color='pink')

    plt.xlabel('Height (cm)')
    plt.ylabel('Frequency')
    plt.title('Height Histogram by Gender')
    plt.legend()

    plt.show()

    #print(y_data, x_data)


if __name__ == "__main__":
    main()