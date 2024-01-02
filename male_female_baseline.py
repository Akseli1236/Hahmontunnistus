import numpy
import random


def main():
    x_data_train = numpy.loadtxt('male_female_X_train.txt')
    y_data_train = numpy.loadtxt('male_female_y_train.txt')

    x_data_test = numpy.loadtxt('male_female_X_test.txt')
    y_data_test = numpy.loadtxt('male_female_y_test.txt')

    y_data_train = y_data_train.astype(numpy.int64)

    random_class = [random.choice(y_data_train) for _ in range(len(x_data_test))]
    correct_count_random = sum(1 for p, t in zip(random_class, y_data_test) if p == t)
    random_accuracy = correct_count_random / len(y_data_test)
    print(f"Random Classifier Accuracy: {random_accuracy:.2%}")

    most_likely_class = numpy.bincount(y_data_train).argmax()
    most_likely_prediction = [most_likely_class] * len(x_data_test)
    correct_count_most_likely = sum(1 for p, t in zip(most_likely_prediction, y_data_test) if p == t)
    most_likely_accuracy = correct_count_most_likely / len(y_data_test)

    print(f"Most Likely Classifier Accuracy: {most_likely_accuracy:.2%}: The class is {most_likely_class}")


if __name__ == "__main__":
    main()