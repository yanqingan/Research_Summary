def find_missing_number(nums):
    n = len(nums)
    # Expected sum of numbers from 0 to n
    expected_sum = n * (n + 1) // 2
    # Actual sum of the given numbers
    actual_sum = sum(nums)
    # The difference is the missing number
    return expected_sum - actual_sum

# Test case
nums = [3, 0, 1]
result = find_missing_number(nums)
print(result)
