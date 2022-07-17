# Def if user input is valid
def is_input_invalid(input):
    possible_args = [2, 3]

    invalid_length = len(input) not in possible_args

    if invalid_length :
        return True
