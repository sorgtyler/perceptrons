# Author: Tyler Sorg
# Course: CS445 Machine Learning
# Project: Perceptron learning algorithm and all-pairs method for classifying english letters.

import os
import itertools
import random
import csv
import sys


def random_weights():
    # Returns a numpy array of 17 random floats between -1 and 1
    # return array([random.uniform(-1, 1) for x in range(17)])
    return [random.uniform(-1, 1) for x in range(17)]


# Initialize dictionary of perceptrons.
def initialize_perceptrons(classifier_names_arg, perceptrons_dictionary_arg):
    for pair in classifier_names_arg:
        # {'AB' = array([0.68417208 -0.24275294 -0.72554846  0.02487458 ...])}
        perceptrons_dictionary_arg[pair[0] + pair[1]] = random_weights()


def initialize_all_perceptrons_accuracies(classifier_names_arg, perceptrons_accuracy):
    for pair in classifier_names_arg:
        # {'AB' = float}
        perceptrons_accuracy[pair[0] + pair[1]] = 0.0


def save_perceptrons(filename, perceptrons_dict):
    # Create new file every time or overwrite old file? one and same?
    # f = open(filename + '.txt', 'w')
    f = open(filename, 'w')
    for i in perceptrons_dict:
        # f.write(i) #if you want "AB,weights"
        f.write(i[0] + ',' + i[1])  # if you want "A,B,weights"
        # weights = perceptrons_dict[i].tolist() #array version
        weights = perceptrons_dict[i]
        for j in weights:
            f.write(',' + str(j))
        f.write('\n')
    f.close()


def save_perceptrons_accuracies(filename, perceptrons_accuracies):
    # Create new file every time or overwrite old file? one and same?
    # f = open(filename + '.txt', 'w')

    # Get name of file without .txt extension
    file_prefix = filename[:len(filename) - 4]
    acc_suffix = '_accuracies'
    extension = '.txt'

    f = open(file_prefix + acc_suffix + extension, 'w')
    for i in perceptrons_accuracies:
        # f.write(i) #if you want "AB,weights"
        f.write(i[0] + ',' + i[1])  # if you want "A,B,weights"
        accuracy = perceptrons_accuracies[i]
        f.write(',' + str(accuracy))
        f.write('\n')
    f.close()


def rebuild_perceptrons(filename):
    """

    :param filename: Name of file to build the dictionary from
    :return: the dictionary parsed from the file.
    """
    # f = open(filename + '.txt', 'r')
    f = open(filename, 'r')
    perceptrons = {}
    for encoding in f:
        # Get rid of newline and then create a list of the values delimited by the comma ,
        encoding = encoding.strip().split(',')
        # Get A,B from string encoding
        name = encoding[0] + encoding[1]  # name = "A" + "B"
        weights = [float(i) for i in encoding[2:]]
        # perceptrons[name] = array(weights)
        perceptrons[name] = weights
    return perceptrons


def rebuild_perceptron_accuracies(filename):
    """

    :param filename: Name of file to build the dictionary from
    :return: the dictionary parsed from the file.
    """
    file_prefix = filename[:len(filename) - 4]
    acc_suffix = '_accuracies'
    extension = '.txt'
    f = open(file_prefix + acc_suffix + extension, 'r')
    accuracies = {}
    for encoding in f:
        # Get rid of newline and then create a list of the values delimited by the comma ,
        encoding = encoding.strip().split(',')
        # Get A,B from string encoding
        name = encoding[0] + encoding[1]  # name = "A" + "B"
        accuracy = float(encoding[2])
        accuracies[name] = accuracy
    return accuracies


def train_perceptrons(perceptrons, perceptron_accuracy, number_of_epochs):
    """

    :param perceptrons: The dictionary of perceptron weights
    :param perceptron_accuracy: The dictionary of perceptron accuracies
    :param number_of_epochs: How many times to iterate over training examples when training.
    """
    training_folder = os.path.join('.', 'combined_training/')

    # For every perceptron
    for p in perceptrons:
        weights = perceptrons[p]

        # Get its training examples.
        examples_filename = p + '.txt'
        examples = open(training_folder + examples_filename, 'r')
        # print 'filename:\n', examples_filename
        reader = csv.reader(examples)

        # Put all training examples for this perceptron in memory using one list.
        all_training_examples = list()
        for row in reader:
            if len(row) > 0:
                all_training_examples.append(row)

        # Train for a certain number of epochs.
        epochs = 0
        while epochs < number_of_epochs:
            epochs += 1

            # Perceptron agrees with target value.
            y_is_target = 0

            # BEGIN EPOCH
            for example in all_training_examples:

                # Determine target class.
                target_class = get_target_class(example)

                # Get the value to compare the perceptron's output to. -1 if Class 1 is correct answer. 1 if Class 2.
                if eligible_perceptron(p, target_class):
                    target_value = get_target_value(p, target_class)
                else:
                    target_value = random_target_value()

                # Get inputs array.
                inputs_list = example[1:]
                inputs = make_inputs_array_with_list(inputs_list)

                # Calculate perceptron output
                y = signum(weights, inputs)

                # Compare y to target value
                if y is target_value:
                    y_is_target += 1
                elif y is not target_value:
                    # update_weights(perceptrons[p], inputs, target_value)
                    update_weights(weights, inputs, target_value)
            # END EPOCH

            accuracy = float(y_is_target) / len(all_training_examples)

            perceptron_accuracy[p] = accuracy
            print 'Perceptron %s accuracy is now: ' % p, accuracy


# Get the perceptron's output based on the weights and inputs dotted together.
def signum(weights, inputs):
    """

    :param weights: Perceptron weights
    :param inputs: Example inputs
    :return: If non-negative dot product, return +1. Else return -1.
    """
    z = 0
    for i in range(17):
        z += weights[i]*inputs[i]
    if z < 0:
        return -1
    elif z >= 0:
        return 1


# Create a standard list with the format ['A', 0.133333, ...]
def parse_training_example(example_string_encoding):
    """

    :param example_string_encoding: Take list of string representations of target class and inputs
    :return: return the floating point versions of the inputs
    """
    example = example_string_encoding.strip().split(',')
    target_class = example[0]
    formatted_example = list()
    formatted_example.append(target_class)
    formatted_example.extend([float(i) for i in example[1:]])
    return formatted_example


# Return 'A' from an example with the format ['A', 0.133333, ...]
def get_target_class(example):
    """

    :param example: Given an example array containing the target class and 16 weights,
    :return: return the target class represented as a string.
    """
    return example[0]


# Example is the slice [1:] from the training examples or just the whole test example based on data encoding.
def make_inputs_array_with_list(example):
    """

    :param example: Given an array of 16 string representations of floats representing a letter
    :return: Transform the example array to add x_0 = 1 and cast the other elements to floats and return it.
    """
    x0 = 1
    result = [x0]
    for x_i in example:
        result.append(float(x_i))

    return result


def eligible_perceptron(perceptron_name, target_class):
    """

    :param perceptron_name: The key in the perceptron dictionary. Used to decide actual letter's target value.
    :param target_class: Actual letter to be classified.
    :return: If the perceptron decides on a relevant letter, return True.
    """
    if target_class in perceptron_name:
        return True
    else:
        return False


# Call this if the target is appropriate for the perceptron.
def get_target_value(perceptron_name, target_class):
    """

    :param perceptron_name: The key in the perceptron dictionary. Used to decide actual letter's target value.
    :param target_class: Actual letter to be classified.
    :return: the target value to compare to the perceptron's output.
    """
    if target_class is perceptron_name[0]:
        return -1
    elif target_class is perceptron_name[1]:
        return 1


# Call this if the perceptron classifies letters not related to the target class.
def random_target_value():
    """

    :return: Return +1 or -1
    """
    return random.randrange(-1, 2, 2)


def update_weights(weights, inputs, target):
    """

    :param weights: Perceptron weights.
    :param inputs: Inputs array
    :param target: Target value (+1 or -1)
    """
    eta = 0.2
    for i in range(17):
        weights[i] += delta_weight_i(eta, inputs[i], target)


def delta_weight_i(eta, input_i, target):
    """

    :param eta: Learning rate
    :param input_i: Input value at index i in the inputs array.
    :param target: The target value (+1 or -1)
    :return: return the product of the three arguments. This is equivalent to Delta(weight_i)
    """
    return eta * input_i * target


def most_voted_letter(inputs, perceptrons):
    # Set up a dictionary of votes
    """

    :param inputs: Given 17 input values in the input arrays and
    :param perceptrons: All of the perceptrons
    :return: The letter/class that the perceptrons most commonly classified the inputs as.
    """
    voted_letter = '?'
    votes = create_letter_table()

    for p in perceptrons:

        # Perceptron output
        weights = perceptrons[p]
        y = signum(weights, inputs)
        # Perceptron says the inputs indicate Class 1
        if y is -1:
            voted_letter = p[0]
        # Perceptron says the inputs indicate Class 2
        elif y is 1:
            voted_letter = p[1]

        votes[voted_letter] += 1
    # Return key of (first) largest value in votes. TODO: Choose value randomly if there is a tie.
    return max(votes, key=votes.get)


def create_letter_table():
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    d = dict()
    for letter in letters:
        d[letter] = 0
    return d


def main():
    # Class names = letters of the alphabet
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # Object that contains ordered pairs of all combinations of the letters without repeats. ('A','B') is ('B','A')
    classifier_names = itertools.combinations(letters, 2)

    # Perceptron 'AB': list of weights
    perceptrons = dict()

    # For recording accuracy of each perceptron after training phase ends.
    perceptron_accuracy = dict()

    # Default filename for saving encodings (and accuracies).
    perceptron_filename = 'perceptron_encodings.txt'

    number_of_epochs = 5
    if len(sys.argv) > 1:
        # Optional hyper-parameter for adjusting number of epochs
        if len(sys.argv) > 2:
            number_of_epochs = int(sys.argv[2])
        perceptron_filename = sys.argv[1]

        try:  # Try building from perceptrons and accuracies files
            # The new filename is the argument
            if '.txt' not in perceptron_filename:  # In case extension not provided
                perceptron_filename += '.txt'

            # Rebuild perceptrons
            perceptrons = rebuild_perceptrons(perceptron_filename)

            # Rebuild perceptrons accuracies
            perceptron_accuracy = rebuild_perceptron_accuracies(perceptron_filename)

        except IOError:  # Couldn't find files, generate new perceptrons and train them.
            # Random weights, need to train.
            initialize_perceptrons(classifier_names, perceptrons)

            # All the accuracies are 0.0% to before training
            initialize_all_perceptrons_accuracies(classifier_names, perceptron_accuracy)

            # Train perceptrons and record accuracies for each after a number of epochs.
            train_perceptrons(perceptrons, perceptron_accuracy, number_of_epochs)

            # Save them to file
            save_perceptrons(perceptron_filename, perceptrons)
            save_perceptrons_accuracies(perceptron_filename, perceptron_accuracy)

    elif len(sys.argv) is 1:  # No argument provided for specifying perceptrons to test. Create and train new ones.
        # Random weights, need to train.
        initialize_perceptrons(classifier_names, perceptrons)
        # All the accuracies are 0.0% to before training
        initialize_all_perceptrons_accuracies(classifier_names, perceptron_accuracy)
        train_perceptrons(perceptrons, perceptron_accuracy, number_of_epochs)
        # Train new perceptrons
        # SAVE PERCEPTRONS
        save_perceptrons(perceptron_filename, perceptrons)
        save_perceptrons_accuracies(perceptron_filename, perceptron_accuracy)

    # ALL-PAIRS
    # Set up a list of ordered pairs ('actual','predicted'); e.g., [('A','A'), ('A','B'), ...]
    confusion_matrix_points = list()
    # Keep track of correctly classified letters by letter.
    votes_by_letter = dict()
    # Keep track of total tests
    global_total = 0
    # Keep track of total correct test classifications
    global_correct = 0

    test_folder = os.path.join('.', 'testing/')
    for letter in letters:
        test_file = open(test_folder + letter + '.txt', 'r')
        test_reader = csv.reader(test_file)

        # For each test in the current letter
        correctly_voted_letter_count = 0
        number_of_tests_for_that_letter = 0

        votes_by_letter[letter] = {'total': 0, 'correct': 0, 'pairs': []}

        for row in test_reader:
            if len(row) > 0:
                # Count the tests
                number_of_tests_for_that_letter += 1

                # Get the input array
                # inputs = make_inputs_array_with_list(row)

                inputs = [1]
                for inp in row:
                    inputs.append(float(inp))

                # Run all the perceptrons on that input array and tally votes.
                voted_letter = most_voted_letter(inputs, perceptrons)

                # Collect all votes
                confusion_matrix_points.append(tuple((letter, voted_letter)))

                # Collect votes for just this letter's statistics.
                votes_by_letter[letter]['pairs'].append(tuple((letter, voted_letter)))

                # Count correct classifications
                if voted_letter is letter:  # correct classification by all-pairs voting
                    correctly_voted_letter_count += 1

        # Record total tests for each letter
        votes_by_letter[letter]['total'] = number_of_tests_for_that_letter
        global_total += number_of_tests_for_that_letter

        # Record total correct classifications for each letter
        votes_by_letter[letter]['correct'] = correctly_voted_letter_count
        global_correct += correctly_voted_letter_count

        total = number_of_tests_for_that_letter
        correct = correctly_voted_letter_count
        test_accuracy = float(correct) / total
        print 'Letter: %s Tests: %d Correct votes: %d Correct/Total: %f' % (letter, total, correct, test_accuracy)
        print 'Testing for %s done\n\n' % letter

    global_accuracy = float(global_correct) / global_total
    print 'Average accuracy: %f' % global_accuracy

    # Create confusion matrix information
    confusion_matrix_data_file = open(
            perceptron_filename[:len(perceptron_filename) - 4] + '_confusion_matrix_pairs' + '.csv', 'w')
    writer = csv.writer(confusion_matrix_data_file, delimiter=',')

    pairs = itertools.product(letters, letters)
    confusion_matrix = dict()
    for pair in pairs:
        confusion_matrix[pair] = 0
    for letter in votes_by_letter:
        for pair in votes_by_letter[letter]['pairs']:
            confusion_matrix[pair] += 1

    for x in letters:
        row = []
        for y in letters:
            row.append(confusion_matrix[(x, y)])
        writer.writerow(row)


main()
