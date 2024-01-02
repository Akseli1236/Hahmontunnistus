import numpy as np

#Used data
x_data_train = np.loadtxt('male_female_X_train.txt')
y_data_train = np.loadtxt('male_female_y_train.txt')

x_data_test = np.loadtxt('male_female_X_test.txt')
y_data_test = np.loadtxt('male_female_y_test.txt')

y_data_train = y_data_train.reshape(-1, 1)

# Combine the data
all_data = np.hstack((x_data_train, y_data_train))
male_height_train = all_data[all_data[:, 2] == 0][:, 0]
female_height_train = all_data[all_data[:, 2] == 1][:, 0]

male_weight_train = all_data[all_data[:, 2] == 0][:, 1]
female_weight_train = all_data[all_data[:, 2] == 1][:, 1]

# Prior for male and female
prior_male = np.mean(y_data_train == 0)
prior_female = np.mean(y_data_train == 1)

print("Prior for male: ", prior_male)
print("Prior for female: ", prior_female)

# Calculate histograms for height and weight for males and females

hist_height_male, bin_edges_height_male = np.histogram(male_height_train)
hist_weight_male, bin_edges_weight_male = np.histogram(male_weight_train)

hist_height_female, bin_edges_height_female = np.histogram(female_height_train)
hist_weight_female, bin_edges_weight_female = np.histogram(female_weight_train)

# Compute the bin centroids for height and weight
bin_centers_height_male = (bin_edges_height_male[:-1] +
                           bin_edges_height_male[1:]) / 2
bin_centers_weight_male = (bin_edges_weight_male[:-1] +
                           bin_edges_weight_male[1:]) / 2

bin_centers_height_female = (bin_edges_height_female[:-1] +
                             bin_edges_height_female[1:]) / 2
bin_centers_weight_female = (bin_edges_weight_female[:-1] +
                             bin_edges_weight_female[1:]) / 2


# Now, for each test sample, find the closest bin and compute likelihood
def compute_likelihood(sample, bin_centers, histogram):
    closest_bin = np.argmin(np.abs(bin_centers - sample))
    likelihood = histogram[closest_bin] / len(x_data_train)
    return likelihood


# Calculate class likelihoods for each test sample
likelihoods_height_male = [
    compute_likelihood(sample, bin_centers_height_male, hist_height_male) for
    sample in x_data_test[:, 0]]
likelihoods_weight_male = [
    compute_likelihood(sample, bin_centers_weight_male, hist_weight_male) for
    sample in x_data_test[:, 1]]

likelihoods_height_female = [
    compute_likelihood(sample, bin_centers_height_female, hist_height_female)
    for sample in x_data_test[:, 0]]
likelihoods_weight_female = [
    compute_likelihood(sample, bin_centers_weight_female, hist_weight_female)
    for sample in x_data_test[:, 1]]

p_height_male_list = []
p_weight_male_list = []
p_height_female_list = []
p_weight_female_list = []

# Gather the likelihoods in a list that represents them
for i in range(len(x_data_test)):
    p_height_male_list.append(likelihoods_height_male[i])
    p_weight_male_list.append(likelihoods_weight_male[i])
    p_height_female_list.append(likelihoods_height_female[i])
    p_weight_female_list.append(likelihoods_weight_female[i])

# Calculate sums of the P for all the likelihoods
sum_height_male = sum(p_height_male_list)
sum_weight_male = sum(p_weight_male_list)
sum_height_female = sum(p_height_female_list)
sum_weight_female = sum(p_weight_female_list)

print(f"Height probability for males: {sum_height_male}")
print(f"Weight probability for males: {sum_weight_male}")
print(f"Height probability for females: {sum_height_female}")
print(f"Weight probability for females: {sum_weight_female}")


# Define a function to classify a test sample based on the likelihoods
def classify_combined(samples):
    # Classify as male (0) or female (1) based on the likelihoods
    if samples[0] * samples[1] > samples[2] * \
            samples[3]:
        return 0  # Male
    else:
        return 1  # Female


# Define functions to classify test samples based on different features
def classify_accuracy(sample_likelihood):
    # Classify as male (0) or female (1) based on height likelihood
    if sample_likelihood[0] > sample_likelihood[1]:
        return 0  # Male
    else:
        return 1  # Female


# Initialize lists to store the predicted labels
predicted_labels = []

# Classify each test sample and store the predicted label
for i in range(len(x_data_test)):
    sample_likelihoods = [likelihoods_height_male[i],
                          likelihoods_weight_male[i],
                          likelihoods_height_female[i],
                          likelihoods_weight_female[i]]
    predicted_label = classify_combined(sample_likelihoods)
    predicted_labels.append(predicted_label)

# Calculate the classification accuracy
correct_predictions = np.sum(predicted_labels == y_data_test)
total_samples = len(x_data_test)
accuracy = correct_predictions / total_samples * 100

print(f"Classification Accuracy: {accuracy:.2f}%")

# Initialize lists to store the predicted labels
predicted_labels_height = []
predicted_labels_weight = []
predicted_labels_combined = []

# Classify each test sample and store the predicted labels
for i in range(len(x_data_test)):
    sample_likelihoods = [likelihoods_height_male[i],
                          likelihoods_height_female[i],
                          likelihoods_weight_male[i],
                          likelihoods_weight_female[i]]

    # Classify based on height likelihood
    predicted_label_height = classify_accuracy(sample_likelihoods[:2])
    predicted_labels_height.append(predicted_label_height)

    # Classify based on weight likelihood
    predicted_label_weight = classify_accuracy(sample_likelihoods[2:])
    predicted_labels_weight.append(predicted_label_weight)

    # Classify based on combined height and weight likelihood
    if sample_likelihoods[0] * sample_likelihoods[2] > sample_likelihoods[1] * \
            sample_likelihoods[3]:
        predicted_labels_combined.append(0)  # Male
    else:
        predicted_labels_combined.append(1)  # Female

# Calculate the classification accuracy for height only, weight only,
# and combined (height and weight)
correct_predictions_height = np.sum(predicted_labels_height == y_data_test)
accuracy_height_only = correct_predictions_height / total_samples * 100

correct_predictions_weight = np.sum(predicted_labels_weight == y_data_test)
accuracy_weight_only = correct_predictions_weight / total_samples * 100

correct_predictions_combined = np.sum(predicted_labels_combined == y_data_test)
accuracy_combined = correct_predictions_combined / total_samples * 100

print(f"Classification Accuracy (Height Only): {accuracy_height_only:.2f}%")
print(f"Classification Accuracy (Weight Only): {accuracy_weight_only:.2f}%")
print(f"Classification Accuracy (Combined Height and Weight): {accuracy_combined:.2f}%")
