import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def gaussian_equation(x, mean, std):
    exp = -0.5 * ((x - mean) / std) ** 2
    return (1 / (std * numpy.sqrt(2 * numpy.pi))) * numpy.exp(exp)


def main():
    x_train = numpy.loadtxt('X_train.txt')[:, 0]
    x_test = numpy.loadtxt('X_test.txt')[:, 0]

    y_train = numpy.loadtxt('y_train.txt')
    y_test = numpy.loadtxt('y_test.txt')

    baseline = numpy.random.randint(0, 2, size=len(y_test))
    accuracy_all = accuracy_score(y_test, baseline)

    accuracy_male = accuracy_score(y_test[y_test == 0], baseline[y_test == 0])
    accuracy_female = accuracy_score(y_test[y_test == 1],
                                     baseline[y_test == 1])

    print("Accuracy:", accuracy_all)
    print("Accuracy for males", accuracy_male)
    print("Accuracy for females", accuracy_female)

    height_range = numpy.linspace(120, 210, 100)

    all_data = numpy.column_stack((x_train, y_train))
    male_height = all_data[all_data[:, 1] == 0][:, 0]
    female_height = all_data[all_data[:, 1] == 1][:, 0]

    male_likelihood_gaussian = [gaussian_equation(height_range,
                                                  numpy.mean(male_height),
                                                  numpy.std(male_height))
                                for i in male_height]

    female_likelihood_gaussian = [gaussian_equation(height_range,
                                                    numpy.mean(female_height),
                                                    numpy.std(female_height))
                                  for i in female_height]

    for i, likelihood in enumerate(male_likelihood_gaussian):
        plt.plot(height_range, likelihood, 'k-')

    for i, likelihood in enumerate(female_likelihood_gaussian):
        plt.plot(height_range, likelihood, 'r-')

    plt.xlabel("Height")
    plt.ylabel("Likelihood")
    plt.title("Likelihoods for Male and Female Samples")
    plt.show()

    prior_male = numpy.mean(y_test == 0)
    prior_female = numpy.mean(y_test == 1)

    predictions = []

    for test_sample in x_test:
        likelihood_male = gaussian_equation(test_sample,
                                            numpy.mean(male_height),
                                            numpy.std(male_height))
        likelihood_female = gaussian_equation(test_sample,
                                              numpy.mean(female_height),
                                              numpy.std(female_height))

        class_male = likelihood_male * prior_male
        class_female = likelihood_female * prior_female

        sum_of_male_and_female = class_male + class_female
        prob_male = class_male / sum_of_male_and_female
        prob_female = class_female / sum_of_male_and_female

        if prob_male > prob_female:
            predictions.append(0)
        else:
            predictions.append(1)

    accuracy = accuracy_score(y_test, predictions)

    print("Correct Classification Percentage:", accuracy)


if __name__ == "__main__":
    main()
