import numpy
import matplotlib.pyplot as plt


def main():
    # Load the data
    x_data_train = numpy.loadtxt('male_female_X_train.txt')
    y_data_train = numpy.loadtxt('male_female_y_train.txt')

    x_data_test = numpy.loadtxt('male_female_X_test.txt')
    y_data_test = numpy.loadtxt('male_female_y_test.txt')

    y_data_train = y_data_train.reshape(-1, 1)

    # Combine the data
    all_data = numpy.hstack((x_data_train, y_data_train))
    male_heigth = all_data[all_data[:, 2] == 0][:, 0]
    female_heigth = all_data[all_data[:, 2] == 1][:, 0]

    male_weight = all_data[all_data[:, 2] == 0][:, 1]
    female_weight = all_data[all_data[:, 2] == 1][:, 1]

    # Create subplot so we can have both histograms in the same picture
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(male_heigth, bins=10, range=(80, 220), alpha=0.6, label='Male',
                color='blue')
    axs[0].hist(female_heigth, bins=10, range=(80, 220), alpha=0.6,
                label='Female', color='pink')

    axs[0].set_xlabel('Height (cm)')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Height Histogram by Gender')
    axs[0].legend()

    axs[1].hist(male_weight, bins=10, range=(30, 180), alpha=0.6, label='Male',
                color='blue')
    axs[1].hist(female_weight, bins=10, range=(30, 180), alpha=0.6,
                label='Female', color='pink')

    axs[1].set_xlabel('Weight (kg)')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Weight Histogram by Gender')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
