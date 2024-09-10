def two_sum(nums, target):
    # the dictionary serves as a hash map
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None

# Test the function with an example
nums = [2, 7, 19, 22, 11, 15, 30]
target = 22
result = two_sum(nums, target)
print(result)