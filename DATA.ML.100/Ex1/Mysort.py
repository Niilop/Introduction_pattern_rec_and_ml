import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')
my_numbers = [None]*len(arr)
for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)

# Print
print(f'Before sorting {my_numbers}')


# My sorting (e.g. bubble sort)

for i in range (0, len(my_numbers)-1):
    changes = False
    print("My numbers are now: ", my_numbers)
    for j in range(0, len(my_numbers) - 1 - i):
        if my_numbers[j] > my_numbers[j+1]:
            my_numbers[j], my_numbers[j+1] = my_numbers[j+1], my_numbers[j]
            changes = True

    if not changes:
        #breaks the loop if no changes were made == the numbers are sorted
        break

# Print
print(f'After sorting {my_numbers}')